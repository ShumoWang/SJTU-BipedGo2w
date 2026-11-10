/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"

#include <functional>

namespace
{
bool IsRobotControlKey(int key)
{
    if (key >= '0' && key <= '9')
    {
        return true;
    }
    switch (key)
    {
    case 'A': case 'D': case 'E': case 'N': case 'P':
    case 'Q': case 'R': case 'S': case 'W': case ' ':
    case '\n':
        return true;
    default:
        return false;
    }
}

class RLGlfwAdapter : public mj::GlfwAdapter
{
public:
    explicit RLGlfwAdapter(std::function<bool(int)> key_handler)
        : key_handler_(std::move(key_handler)) {}

protected:
    void OnKey(int key, int scancode, int action) override
    {
        if (IsKeyDownEvent(action))
        {
            int translated_key = TranslateKeyCode(key);
            if (translated_key >= GLFW_KEY_KP_0 && translated_key <= GLFW_KEY_KP_9)
            {
                translated_key = '0' + translated_key - GLFW_KEY_KP_0;
            }
            else if (translated_key == GLFW_KEY_ENTER || translated_key == GLFW_KEY_KP_ENTER)
            {
                translated_key = '\n';
            }

            if (IsRobotControlKey(translated_key) && key_handler_(translated_key))
            {
                return;
            }
        }
        mj::GlfwAdapter::OnKey(key, scancode, action);
    }

private:
    std::function<bool(int)> key_handler_;
};
}

RL_Sim* RL_Sim::instance = nullptr;

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name [policy_trace.csv]" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }

    if (argc >= 4)
    {
        this->policy_trace_path = argv[3];
    }
    else if (const char* trace_path = std::getenv("RL_SAR_POLICY_TRACE"))
    {
        this->policy_trace_path = trace_path;
    }
    if (this->policy_trace_path.empty())
    {
        std::cout << LOGGER::INFO << "MuJoCo policy trace disabled" << std::endl;
    }
    else
    {
        std::cout << LOGGER::INFO << "MuJoCo policy trace armed: " << this->policy_trace_path
                  << " (file is created when policy inference starts)" << std::endl;
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<RLGlfwAdapter>([this](int key) { return this->HandleKeyboardKey(key); }),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    this->base_body_id = mj_name2id(this->mj_model, mjOBJ_BODY, "base_link");
    this->object_body_id = mj_name2id(this->mj_model, mjOBJ_BODY, "CarryCube");
    this->object_joint_id = mj_name2id(this->mj_model, mjOBJ_JOINT, "CarryCube_freejoint");
    this->SetupSysJoystick("/dev/input/js0", 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
    this->loop_control->start();
    this->loop_rl->start();

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    this->loop_rl->shutdown();
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        if (this->base_body_id >= 0)
        {
            const mjtNum* base_position = this->mj_data->xpos + 3 * this->base_body_id;
            state->imu.position = {
                static_cast<float>(base_position[0]),
                static_cast<float>(base_position[1]),
                static_cast<float>(base_position[2])
            };

            mjtNum base_velocity[6] = {0.0};
            mj_objectVelocity(this->mj_model, this->mj_data, mjOBJ_BODY, this->base_body_id, base_velocity, 1);
            state->imu.linear_velocity = {
                static_cast<float>(base_velocity[3]),
                static_cast<float>(base_velocity[4]),
                static_cast<float>(base_velocity[5])
            };
        }

        state->imu.quaternion[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 0];
        state->imu.quaternion[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 1];
        state->imu.quaternion[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 2];
        state->imu.quaternion[3] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 3];

        state->imu.gyroscope[0] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 4];
        state->imu.gyroscope[1] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 5];
        state->imu.gyroscope[2] = mj_data->sensordata[3 * this->params.Get<int>("num_of_dofs") + 6];

        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            state->motor_state.q[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]];
            state->motor_state.dq[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")];
            state->motor_state.tau_est[i] = mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + 2 * this->params.Get<int>("num_of_dofs")];
        }

        if (this->object_body_id >= 0)
        {
            const mjtNum* object_position = this->mj_data->xpos + 3 * this->object_body_id;
            const mjtNum* object_quaternion = this->mj_data->xquat + 4 * this->object_body_id;
            state->object_state.position = {
                static_cast<float>(object_position[0]),
                static_cast<float>(object_position[1]),
                static_cast<float>(object_position[2])
            };
            state->object_state.quaternion = {
                static_cast<float>(object_quaternion[0]),
                static_cast<float>(object_quaternion[1]),
                static_cast<float>(object_quaternion[2]),
                static_cast<float>(object_quaternion[3])
            };

            mjtNum object_velocity[6] = {0.0};
            mj_objectVelocity(this->mj_model, this->mj_data, mjOBJ_BODY, this->object_body_id, object_velocity, 0);
            state->object_state.angular_velocity = {
                static_cast<float>(object_velocity[0]),
                static_cast<float>(object_velocity[1]),
                static_cast<float>(object_velocity[2])
            };
            state->object_state.linear_velocity = {
                static_cast<float>(object_velocity[3]),
                static_cast<float>(object_velocity[4]),
                static_cast<float>(object_velocity[5])
            };
        }
    }
}

bool RL_Sim::ResetHOIState()
{
    if (!this->mj_model || !this->mj_data || !this->hoi_motion_loader)
    {
        return false;
    }
    if (this->base_body_id < 0 || this->object_body_id < 0 || this->object_joint_id < 0)
    {
        std::cout << LOGGER::ERROR << "HOI scene must contain base_link, CarryCube, and CarryCube_freejoint" << std::endl;
        return false;
    }
    const auto joint_names = this->params.Get<std::vector<std::string>>("mujoco_joint_names");
    const auto joint_positions = this->hoi_motion_loader->GetJointPos();
    const auto joint_velocities = this->hoi_motion_loader->GetJointVel();
    if (joint_names.size() != joint_positions.size() || joint_positions.size() != joint_velocities.size())
    {
        std::cout << LOGGER::ERROR << "HOI motion and mujoco_joint_names have different sizes" << std::endl;
        return false;
    }

    mj_resetData(this->mj_model, this->mj_data);

    const int base_joint_id = this->mj_model->body_jntadr[this->base_body_id];
    if (base_joint_id < 0 || this->mj_model->jnt_type[base_joint_id] != mjJNT_FREE)
    {
        std::cout << LOGGER::ERROR << "base_link must have a free joint" << std::endl;
        return false;
    }

    const int base_qpos_address = this->mj_model->jnt_qposadr[base_joint_id];
    const int base_dof_address = this->mj_model->jnt_dofadr[base_joint_id];
    const auto anchor_position = this->hoi_motion_loader->GetAnchorPos();
    const auto anchor_quaternion = QuaternionNormalize(this->hoi_motion_loader->GetAnchorQuat());
    const auto anchor_linear_velocity = this->hoi_motion_loader->GetAnchorLinVel();
    const auto anchor_angular_velocity_body = QuatRotateInverse(
        anchor_quaternion,
        this->hoi_motion_loader->GetAnchorAngVel()
    );
    for (int i = 0; i < 3; ++i)
    {
        this->mj_data->qpos[base_qpos_address + i] = anchor_position[i];
        this->mj_data->qvel[base_dof_address + i] = anchor_linear_velocity[i];
        this->mj_data->qvel[base_dof_address + 3 + i] = anchor_angular_velocity_body[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        this->mj_data->qpos[base_qpos_address + 3 + i] = anchor_quaternion[i];
    }

    for (size_t i = 0; i < joint_names.size(); ++i)
    {
        const int joint_id = mj_name2id(this->mj_model, mjOBJ_JOINT, joint_names[i].c_str());
        if (joint_id < 0)
        {
            std::cout << LOGGER::ERROR << "MuJoCo joint not found: " << joint_names[i] << std::endl;
            return false;
        }
        float reset_joint_position = joint_positions[i];
        if (this->mj_model->jnt_limited[joint_id])
        {
            const float lower = static_cast<float>(this->mj_model->jnt_range[2 * joint_id]);
            const float upper = static_cast<float>(this->mj_model->jnt_range[2 * joint_id + 1]);
            const float midpoint = 0.5f * (lower + upper);
            const float soft_half_range = 0.5f * (upper - lower) *
                                          this->params.Get<float>("soft_joint_pos_limit_factor");
            reset_joint_position = std::clamp(
                reset_joint_position,
                midpoint - soft_half_range,
                midpoint + soft_half_range
            );
        }
        this->mj_data->qpos[this->mj_model->jnt_qposadr[joint_id]] = reset_joint_position;
        this->mj_data->qvel[this->mj_model->jnt_dofadr[joint_id]] = joint_velocities[i];
    }

    const int object_qpos_address = this->mj_model->jnt_qposadr[this->object_joint_id];
    const int object_dof_address = this->mj_model->jnt_dofadr[this->object_joint_id];
    const auto object_position = this->hoi_motion_loader->GetObjectPos();
    const auto object_quaternion = QuaternionNormalize(this->hoi_motion_loader->GetObjectQuat());
    const auto object_linear_velocity = this->hoi_motion_loader->GetObjectLinVel();
    const auto object_angular_velocity_body = QuatRotateInverse(
        object_quaternion,
        this->hoi_motion_loader->GetObjectAngVel()
    );
    for (int i = 0; i < 3; ++i)
    {
        this->mj_data->qpos[object_qpos_address + i] = object_position[i];
        this->mj_data->qvel[object_dof_address + i] = object_linear_velocity[i];
        this->mj_data->qvel[object_dof_address + 3 + i] = object_angular_velocity_body[i];
    }
    for (int i = 0; i < 4; ++i)
    {
        this->mj_data->qpos[object_qpos_address + 3 + i] = object_quaternion[i];
    }

    std::fill(this->mj_data->ctrl, this->mj_data->ctrl + this->mj_model->nu, 0.0);
    mj_forward(this->mj_model, this->mj_data);
    this->episode_length_buf = 0;
    this->obs.actions.assign(this->params.Get<int>("num_of_dofs"), 0.0f);

    std::vector<float> queued_output;
    while (this->output_dof_pos_queue.try_pop(queued_output)) {}
    while (this->output_dof_vel_queue.try_pop(queued_output)) {}
    while (this->output_dof_tau_queue.try_pop(queued_output)) {}

    this->GetState(&this->robot_state);
    this->obs.base_pos = this->robot_state.imu.position;
    this->obs.lin_vel = this->robot_state.imu.linear_velocity;
    this->obs.ang_vel = this->robot_state.imu.gyroscope;
    this->obs.base_quat = this->robot_state.imu.quaternion;
    this->obs.dof_pos = this->robot_state.motor_state.q;
    this->obs.dof_vel = this->robot_state.motor_state.dq;
    this->obs.object_pos = this->robot_state.object_state.position;
    this->obs.object_quat = this->robot_state.object_state.quaternion;
    this->obs.actions = this->Forward();
    this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->robot_command.motor_command.q[i] = this->output_dof_pos[i];
        this->robot_command.motor_command.dq[i] = this->output_dof_vel[i];
        this->robot_command.motor_command.kp[i] = this->params.Get<std::vector<float>>("rl_kp")[i];
        this->robot_command.motor_command.kd[i] = this->params.Get<std::vector<float>>("rl_kd")[i];
        this->robot_command.motor_command.tau[i] = 0.0f;
    }
    this->output_dof_pos_queue.push(this->output_dof_pos);
    this->output_dof_vel_queue.push(this->output_dof_vel);
    this->output_dof_tau_queue.push(this->output_dof_tau);
    this->episode_length_buf = 1;
    std::cout << LOGGER::INFO << "Reset robot and CarryCube to HOI motion start" << std::endl;
    return true;
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            mj_data->ctrl[this->params.Get<std::vector<int>>("joint_mapping")[i]] =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i]]) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - mj_data->sensordata[this->params.Get<std::vector<int>>("joint_mapping")[i] + this->params.Get<int>("num_of_dofs")]);
        }
    }
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->hoi_motion_loader)
        {
            this->hoi_motion_loader->Update(0.0f);
            this->ResetHOIState();
        }
        else if (this->mj_model && this->mj_data)
        {
            mj_resetData(this->mj_model, this->mj_data);
            mj_forward(this->mj_model, this->mj_data);
        }
    }

    this->GetState(&this->robot_state);

    this->StateController(&this->robot_state, &this->robot_command);

    if (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X)
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    this->control.ClearInput();

    this->SetCommand(&this->robot_command);
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::GetSysJoystick()
{
    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            this->sys_js_button[this->sys_js_event.number].update(this->sys_js_event.value);
        }
        else if (this->sys_js_event.isAxis())
        {
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[this->sys_js_event.number] = 0;
            }
            else
            {
                this->sys_js_axis[this->sys_js_event.number] = this->sys_js_event.value;
            }
        }
    }

    if (this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[4].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[4].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[4].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[4].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[4].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[4].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[4].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[5].pressed && this->sys_js_button[0].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[5].pressed && this->sys_js_button[1].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[5].pressed && this->sys_js_button[2].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[5].pressed && this->sys_js_button[3].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[5].pressed && this->sys_js_button[9].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[5].pressed && this->sys_js_button[10].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[7] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[5].pressed && this->sys_js_axis[6] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[4].pressed && this->sys_js_button[5].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float lx = -float(this->sys_js_axis[0]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || lx != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.y = lx;
        this->control.yaw = rx;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->rl_init_done && simulation_running)
    {
        this->episode_length_buf += 1;
        this->obs.base_pos = this->robot_state.imu.position;
        this->obs.lin_vel = this->robot_state.imu.linear_velocity;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        this->obs.commands[0] = this->control.x;
        this->obs.commands[1] = this->control.y;
        this->obs.commands[2] = this->control.yaw;
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands[0] = (float)this->cmd_vel.linear.x;
        //     this->obs.commands[1] = (float)this->cmd_vel.linear.y;
        //     this->obs.commands[2] = (float)this->cmd_vel.angular.z;
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;
        this->obs.object_pos = this->robot_state.object_state.position;
        this->obs.object_quat = this->robot_state.object_state.quaternion;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        actions = clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }

    this->WritePolicyTrace(clamped_obs, actions);
    return actions;
}

void RL_Sim::WritePolicyTrace(const std::vector<float>& observations, const std::vector<float>& actions)
{
    if (!this->policy_trace.is_open())
    {
        if (this->policy_trace_path.empty())
        {
            return;
        }

        const std::filesystem::path trace_path(this->policy_trace_path);
        if (trace_path.has_parent_path())
        {
            std::filesystem::create_directories(trace_path.parent_path());
        }
        this->policy_trace.open(trace_path, std::ios::out | std::ios::trunc);
        if (!this->policy_trace)
        {
            std::cout << LOGGER::ERROR << "Unable to open policy trace: " << this->policy_trace_path << std::endl;
            this->policy_trace_path.clear();
            return;
        }

        this->policy_trace << "frame,time";
        for (size_t i = 0; i < observations.size(); ++i)
        {
            this->policy_trace << ",obs_" << i;
        }
        for (size_t i = 0; i < actions.size(); ++i)
        {
            this->policy_trace << ",action_" << i;
        }
        this->policy_trace << '\n';
        std::cout << LOGGER::INFO << "Writing MuJoCo policy trace to: " << this->policy_trace_path << std::endl;
    }

    const float policy_dt = this->params.Get<float>("dt") * this->params.Get<int>("decimation");
    this->policy_trace << this->policy_trace_frame << ',' << this->policy_trace_frame * policy_dt;
    this->policy_trace.precision(9);
    for (float value : observations)
    {
        this->policy_trace << ',' << value;
    }
    for (float value : actions)
    {
        this->policy_trace << ',' << value;
    }
    this->policy_trace << '\n';
    this->policy_trace.flush();
    ++this->policy_trace_frame;
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
