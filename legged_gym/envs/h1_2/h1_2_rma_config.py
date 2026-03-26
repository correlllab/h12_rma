from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class H1_2RMACfg(LeggedRobotCfg):
    """H1-2 RMA config: RMA-enabled environment with torso + hand force sampling."""

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # -------- Legs (12) -------
            'left_hip_yaw_joint': 0.0,
            'left_hip_pitch_joint': -0.16,
            'left_hip_roll_joint': 0.0,
            'left_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,

            'right_hip_yaw_joint': 0.0,
            'right_hip_pitch_joint': -0.16,
            'right_hip_roll_joint': 0.0,
            'right_knee_joint': 0.36,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,

            # -------- Waist (1) --------
            'torso_joint': 0.0,

            # -------- Left Arm (7) --------
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.0,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_pitch_joint': 0.0,
            'left_elbow_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,

            # -------- Right Arm (7) --------
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': 0.0,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_pitch_joint': 0.0,
            'right_elbow_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }

    class env(LeggedRobotCfg.env):
        # Full-body with hands: 27 actuated DOFs (12 legs + 1 torso + 14 arms)
        # Observation includes proprioception: 3 (ang_vel) + 3 (gravity) + 3 (cmd) + 27 (dof_pos) + 27 (dof_vel) + 27 (action) + 2 (phase) = 92
        num_observations = 92
        num_privileged_obs = 92
        num_actions = 27

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
            'hip_yaw': 200,
            'hip_roll': 200,
            'hip_pitch': 200,
            'knee': 300,
            'ankle_pitch': 60,
            'ankle_roll': 40,
            'torso': 600,
            'shoulder_pitch': 80,
            'shoulder_roll': 80,
            'shoulder_yaw': 40,
            'wrist': 60,
            'elbow': 40,
        }
        damping = {
            'hip_yaw': 5.,
            'hip_roll': 5.,
            'hip_pitch': 5.,
            'knee': 7.5,
            'ankle_pitch': 1.,
            'ankle_roll': 0.3,
            'torso': 15,
            'shoulder_pitch': 2.,
            'shoulder_roll': 2.,
            'shoulder_yaw': 1.,
            'wrist': 0.5,
            'elbow': 1,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/h1_2/h1_2_full.urdf"
        name = "h1_2"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["torso"]
        self_collides = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        action_scales = dict()

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1
        noise_freqs = dict(
            lin_vel=2.0,
            ang_vel=2.0,
            dof_pos=20.0,
            dof_vel=20.0,
            gravity=2.,
            height_measurements = 2.0
        )


class H1_2RMACfgPPO(LeggedRobotCfgPPO):
    """PPO config for H1-2 RMA training."""
    
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3  # a6 was 5.e-3
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_transitions_per_env * num_envs / num_mini_batches
        num_transitions_per_env = 32  # steps_per_env = num * decimation
        lambda_ = 0.95  # advantage discounting
        num_env_steps_per_train_step = 512  # = num_transitions_per_env * num_envs
        policy_lr_schedule = 'adaptive'  # could be adaptive, fixed
        estimator_lr_schedule = 'fixed'

    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # per iteration
        max_iterations = 1500  # number of policy updates

        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'h1_2_rma'
        run_name = f'{experiment_name}'

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated by the command line '---resume_path'
