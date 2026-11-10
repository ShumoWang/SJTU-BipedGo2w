/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "motion_loader.hpp"

MotionLoader::MotionLoader(const std::string& motion_file, float fps)
    : dt_(1.0f / fps), index_0_(0), index_1_(0), blend_(0.0f)
{
    LoadFromCSV(motion_file);
    ComputeVelocities();

    num_frames_ = root_positions_.size();
    duration_ = num_frames_ * dt_;

    std::cout << LOGGER::INFO << "MotionLoader: Loaded " << num_frames_ << " frames, "
              << num_joints_ << " joints, duration=" << duration_ << "s" << std::endl;
}

void MotionLoader::LoadFromCSV(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open motion file: " + filename);
    }

    std::string line;
    bool first_row = true;

    while (std::getline(file, line))
    {
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ','))
        {
            try
            {
                row.push_back(std::stof(value));
            }
            catch (const std::invalid_argument& e)
            {
                std::cerr << "Warning: Invalid value '" << value << "' in CSV" << std::endl;
                row.push_back(0.0f);
            }
        }

        if (row.size() < 7)  // At least root_pos(3) + root_quat(4)
        {
            std::cerr << "Warning: Row has insufficient data, skipping" << std::endl;
            continue;
        }

        // Parse root position: x, y, z
        std::vector<float> root_pos = {row[0], row[1], row[2]};
        root_positions_.push_back(root_pos);

        // Parse root quaternion: x, y, z, w -> w, x, y, z
        std::vector<float> root_quat = {row[6], row[3], row[4], row[5]};
        root_quaternions_.push_back(root_quat);

        // Parse joint positions
        std::vector<float> joint_pos(row.begin() + 7, row.end());
        joint_positions_.push_back(joint_pos);

        if (first_row)
        {
            num_joints_ = joint_pos.size();
            first_row = false;
        }
    }

    file.close();

    if (root_positions_.empty())
    {
        throw std::runtime_error("No valid motion data loaded from: " + filename);
    }
}

void MotionLoader::ComputeVelocities()
{
    joint_velocities_.clear();

    for (size_t i = 0; i < joint_positions_.size(); ++i)
    {
        if (i < joint_positions_.size() - 1)
        {
            // Forward difference
            std::vector<float> vel;
            for (size_t j = 0; j < joint_positions_[i].size(); ++j)
            {
                float dq = (joint_positions_[i + 1][j] - joint_positions_[i][j]) / dt_;
                vel.push_back(dq);
            }
            joint_velocities_.push_back(vel);
        }
        else
        {
            // For last frame, reuse previous velocity
            joint_velocities_.push_back(joint_velocities_.back());
        }
    }
}

void MotionLoader::Update(float time)
{
    // Clamp time to valid range
    float phase = std::clamp(time / duration_, 0.0f, 1.0f);

    // Compute frame indices
    float frame_float = phase * (num_frames_ - 1);
    index_0_ = static_cast<int>(std::floor(frame_float));
    index_1_ = std::min(index_0_ + 1, num_frames_ - 1);

    // Compute blend factor
    blend_ = frame_float - index_0_;
}

void MotionLoader::Reset(const std::vector<float>& robot_base_quat, const std::vector<float>& robot_waist_angles)
{
    Update(0.0f);

    std::vector<float> robot_torso = ComputeTorsoQuat(robot_base_quat, robot_waist_angles);
    std::vector<float> motion_torso = GetAnchorQuat();
    world_to_init_ = ComputeYawAlignment(robot_torso, motion_torso);

    std::cout << LOGGER::INFO << "Motion reset with yaw alignment" << std::endl;
}

std::vector<float> MotionLoader::GetJointPos() const
{
    std::vector<float> result;
    const auto& pos0 = joint_positions_[index_0_];
    const auto& pos1 = joint_positions_[index_1_];

    for (size_t i = 0; i < pos0.size(); ++i)
    {
        result.push_back(pos0[i] * (1.0f - blend_) + pos1[i] * blend_);
    }

    return result;
}

std::vector<float> MotionLoader::GetJointVel() const
{
    std::vector<float> result;
    const auto& vel0 = joint_velocities_[index_0_];
    const auto& vel1 = joint_velocities_[index_1_];

    for (size_t i = 0; i < vel0.size(); ++i)
    {
        result.push_back(vel0[i] * (1.0f - blend_) + vel1[i] * blend_);
    }

    return result;
}

std::vector<float> MotionLoader::GetRootQuat() const
{
    std::vector<float> q = Slerp(root_quaternions_[index_0_], root_quaternions_[index_1_], blend_);
    return q;
}

std::vector<float> MotionLoader::ComputeTorsoQuat(const std::vector<float>& base_quat, const std::vector<float>& waist_angles)
{
    std::vector<float> q_yaw = QuaternionFromAxisAngle({0.0f, 0.0f, 1.0f}, waist_angles[0]);
    std::vector<float> q_roll = QuaternionFromAxisAngle({1.0f, 0.0f, 0.0f}, waist_angles[1]);
    std::vector<float> q_pitch = QuaternionFromAxisAngle({0.0f, 1.0f, 0.0f}, waist_angles[2]);

    std::vector<float> torso_quat = QuaternionMultiply(base_quat, q_yaw);
    torso_quat = QuaternionMultiply(torso_quat, q_roll);
    torso_quat = QuaternionMultiply(torso_quat, q_pitch);

    return QuaternionNormalize(torso_quat);
}

std::vector<float> MotionLoader::ComputeYawAlignment(const std::vector<float>& robot_torso_quat, const std::vector<float>& motion_torso_quat)
{
    std::vector<float> robot_yaw = QuaternionYawOnly(robot_torso_quat);
    std::vector<float> motion_yaw = QuaternionYawOnly(motion_torso_quat);
    return QuaternionMultiply(robot_yaw, QuaternionConjugate(motion_yaw));
}

std::vector<float> MotionLoader::GetAnchorQuat() const
{
    std::vector<float> root_quat = Slerp(root_quaternions_[index_0_], root_quaternions_[index_1_], blend_);
    auto joint_pos = GetJointPos();

    const int WAIST_YAW_IDX = 12;
    const int WAIST_ROLL_IDX = 13;
    const int WAIST_PITCH_IDX = 14;

    std::vector<float> waist_angles = {joint_pos[WAIST_YAW_IDX], joint_pos[WAIST_ROLL_IDX], joint_pos[WAIST_PITCH_IDX]};
    return ComputeTorsoQuat(root_quat, waist_angles);
}

std::vector<float> MotionLoader::Slerp(const std::vector<float>& q0, const std::vector<float>& q1, float t) const
{
    // Compute dot product
    float dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];

    // If dot < 0, negate q1 to take shorter path
    std::vector<float> q1_adjusted = q1;
    if (dot < 0.0f)
    {
        q1_adjusted = {-q1[0], -q1[1], -q1[2], -q1[3]};
        dot = -dot;
    }

    // If quaternions are very close, use linear interpolation
    if (dot > 0.9995f)
    {
        return QuaternionNormalize({
            q0[0] + t * (q1_adjusted[0] - q0[0]),
            q0[1] + t * (q1_adjusted[1] - q0[1]),
            q0[2] + t * (q1_adjusted[2] - q0[2]),
            q0[3] + t * (q1_adjusted[3] - q0[3])
        });
    }

    // Spherical interpolation
    float theta = std::acos(std::clamp(dot, -1.0f, 1.0f));
    float sin_theta = std::sin(theta);

    float w0 = std::sin((1.0f - t) * theta) / sin_theta;
    float w1 = std::sin(t * theta) / sin_theta;

    return {
        q0[0] * w0 + q1_adjusted[0] * w1,
        q0[1] * w0 + q1_adjusted[1] * w1,
        q0[2] * w0 + q1_adjusted[2] * w1,
        q0[3] * w0 + q1_adjusted[3] * w1
    };
}

HoiMotionLoader::HoiMotionLoader(const std::string& motion_file, float fps)
{
    if (fps <= 0.0f)
    {
        throw std::invalid_argument("HOI motion fps must be positive");
    }
    dt_ = 1.0f / fps;

    LoadFromCSV(motion_file);
    num_frames_ = static_cast<int>(joint_positions_.size());
    duration_ = (num_frames_ > 1) ? (num_frames_ - 1) * dt_ : 0.0f;

    std::cout << LOGGER::INFO << "HoiMotionLoader: Loaded " << num_frames_ << " frames, "
              << num_joints_ << " joints, duration=" << duration_ << "s" << std::endl;
}

void HoiMotionLoader::LoadFromCSV(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open HOI motion file: " + filename);
    }

    std::string line;
    size_t line_number = 0;
    while (std::getline(file, line))
    {
        ++line_number;
        if (line.empty())
        {
            continue;
        }

        std::vector<float> row;
        std::stringstream stream(line);
        std::string value;
        while (std::getline(stream, value, ','))
        {
            try
            {
                row.push_back(std::stof(value));
            }
            catch (const std::exception&)
            {
                throw std::runtime_error("Invalid value in HOI motion CSV at line " + std::to_string(line_number));
            }
        }

        constexpr size_t fixed_fields = 26;
        if (row.size() <= fixed_fields || (row.size() - fixed_fields) % 2 != 0)
        {
            throw std::runtime_error("Invalid HOI motion CSV width at line " + std::to_string(line_number));
        }

        const int row_num_joints = static_cast<int>((row.size() - fixed_fields) / 2);
        if (num_joints_ == 0)
        {
            num_joints_ = row_num_joints;
        }
        else if (row_num_joints != num_joints_)
        {
            throw std::runtime_error("Inconsistent joint count in HOI motion CSV at line " + std::to_string(line_number));
        }

        size_t offset = 0;
        auto take = [&row, &offset](size_t count)
        {
            std::vector<float> result(row.begin() + offset, row.begin() + offset + count);
            offset += count;
            return result;
        };

        joint_positions_.push_back(take(num_joints_));
        joint_velocities_.push_back(take(num_joints_));
        anchor_positions_.push_back(take(3));
        anchor_quaternions_.push_back(take(4));
        anchor_linear_velocities_.push_back(take(3));
        anchor_angular_velocities_.push_back(take(3));
        object_positions_.push_back(take(3));
        object_quaternions_.push_back(take(4));
        object_linear_velocities_.push_back(take(3));
        object_angular_velocities_.push_back(take(3));
    }

    if (joint_positions_.empty())
    {
        throw std::runtime_error("No valid HOI motion data loaded from: " + filename);
    }
}

void HoiMotionLoader::Update(float time)
{
    if (num_frames_ <= 1 || duration_ <= 0.0f)
    {
        index_0_ = 0;
        index_1_ = 0;
        blend_ = 0.0f;
        return;
    }

    const float clamped_time = std::clamp(time, 0.0f, duration_);
    const float frame = clamped_time / dt_;
    index_0_ = std::min(static_cast<int>(std::floor(frame)), num_frames_ - 1);
    index_1_ = std::min(index_0_ + 1, num_frames_ - 1);
    blend_ = frame - index_0_;
}

std::vector<float> HoiMotionLoader::Interpolate(const std::vector<std::vector<float>>& values) const
{
    std::vector<float> result(values[index_0_].size(), 0.0f);
    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i] = values[index_0_][i] * (1.0f - blend_) + values[index_1_][i] * blend_;
    }
    return result;
}

std::vector<float> HoiMotionLoader::Slerp(const std::vector<std::vector<float>>& values) const
{
    const auto& q0 = values[index_0_];
    std::vector<float> q1 = values[index_1_];
    float dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
    if (dot < 0.0f)
    {
        for (float& value : q1)
        {
            value = -value;
        }
        dot = -dot;
    }

    if (dot > 0.9995f)
    {
        return QuaternionNormalize({
            q0[0] + blend_ * (q1[0] - q0[0]),
            q0[1] + blend_ * (q1[1] - q0[1]),
            q0[2] + blend_ * (q1[2] - q0[2]),
            q0[3] + blend_ * (q1[3] - q0[3])
        });
    }

    const float theta = std::acos(std::clamp(dot, -1.0f, 1.0f));
    const float sin_theta = std::sin(theta);
    const float weight_0 = std::sin((1.0f - blend_) * theta) / sin_theta;
    const float weight_1 = std::sin(blend_ * theta) / sin_theta;
    return QuaternionNormalize({
        q0[0] * weight_0 + q1[0] * weight_1,
        q0[1] * weight_0 + q1[1] * weight_1,
        q0[2] * weight_0 + q1[2] * weight_1,
        q0[3] * weight_0 + q1[3] * weight_1
    });
}

std::vector<float> HoiMotionLoader::GetJointPos() const { return Interpolate(joint_positions_); }
std::vector<float> HoiMotionLoader::GetJointVel() const { return Interpolate(joint_velocities_); }
std::vector<float> HoiMotionLoader::GetAnchorPos() const { return Interpolate(anchor_positions_); }
std::vector<float> HoiMotionLoader::GetAnchorQuat() const { return Slerp(anchor_quaternions_); }
std::vector<float> HoiMotionLoader::GetAnchorLinVel() const { return Interpolate(anchor_linear_velocities_); }
std::vector<float> HoiMotionLoader::GetAnchorAngVel() const { return Interpolate(anchor_angular_velocities_); }
std::vector<float> HoiMotionLoader::GetObjectPos() const { return Interpolate(object_positions_); }
std::vector<float> HoiMotionLoader::GetObjectQuat() const { return Slerp(object_quaternions_); }
std::vector<float> HoiMotionLoader::GetObjectLinVel() const { return Interpolate(object_linear_velocities_); }
std::vector<float> HoiMotionLoader::GetObjectAngVel() const { return Interpolate(object_angular_velocities_); }

int HoiMotionLoader::FutureFrameIndex(float time_offset) const
{
    if (num_frames_ <= 1 || dt_ <= 0.0f)
    {
        return 0;
    }
    const int frame_offset = static_cast<int>(std::lround(time_offset / dt_));
    return std::clamp(index_0_ + frame_offset, 0, num_frames_ - 1);
}

std::vector<float> HoiMotionLoader::GetFutureJointPos(float time_offset) const
{
    return joint_positions_[FutureFrameIndex(time_offset)];
}

std::vector<float> HoiMotionLoader::GetFutureJointVel(float time_offset) const
{
    return joint_velocities_[FutureFrameIndex(time_offset)];
}

std::vector<float> HoiMotionLoader::GetFutureAnchorPos(float time_offset) const
{
    return anchor_positions_[FutureFrameIndex(time_offset)];
}

std::vector<float> HoiMotionLoader::GetFutureAnchorQuat(float time_offset) const
{
    return anchor_quaternions_[FutureFrameIndex(time_offset)];
}

std::vector<float> HoiMotionLoader::GetFutureObjectPos(float time_offset) const
{
    return object_positions_[FutureFrameIndex(time_offset)];
}

std::vector<float> HoiMotionLoader::GetFutureObjectQuat(float time_offset) const
{
    return object_quaternions_[FutureFrameIndex(time_offset)];
}
