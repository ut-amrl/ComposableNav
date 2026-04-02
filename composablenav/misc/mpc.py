import casadi as ca
from omegaconf import DictConfig
import numpy as np
import time

from composablenav.misc.common import  forward_motion_rollout

def forward_motion_rollout_casadi(state, control, planning_dt):
    # Unpack state and control
    x, y, theta = state[0, :], state[1, :], state[2, :]
    v, omega = control[0, :], control[1, :]

    # Direct computation
    theta_new = theta + omega * planning_dt
    theta_estimate = theta + omega * planning_dt / 2
    x_new = x + v * planning_dt * ca.cos(theta_estimate)
    y_new = y + v * planning_dt * ca.sin(theta_estimate)

    # Combine into result
    return ca.vertcat(x_new, y_new, theta_new)


def mpc_path(cfg: DictConfig, reference_path, planning_dt, start_pos, start_speed, planning_horizon):
    start = time.time()
    # Set buffer in config later
    
    max_linear_accel = cfg.max_dv * planning_dt
    max_angular_accel = cfg.max_dw * planning_dt
    v_max = cfg.max_v
    v_min = 0
    omega_max = cfg.max_w
    omega_min = -cfg.max_w

    # State variables
    state = ca.MX.sym("state", 3)  # [x, y, theta]
    control = ca.MX.sym("control", 2)  # [v, omega]

    # Robot dynamics model
    next_state = forward_motion_rollout_casadi(state, control, planning_dt)

    f = ca.Function("f", [state, control], [next_state])
    
    N = planning_horizon - 1 # Number of control intervals
    # Create lists of state and control variables for the horizon
    X = ca.MX.sym("X", 3, N+1)  # State trajectory
    U = ca.MX.sym("U", 2, N)    # Control trajectory
    
    # Constraints
    start_state = ca.DM(start_pos)
    start_control = ca.DM(start_speed)
    
    # Convert reference_path into a CasADi matrix and vectorize
    reference_path_matrix = ca.DM(reference_path).T  # Shape: 2xN

    # Compute the cost function directly using the reference matrix
    cost = ca.sumsqr(X[:2, 1:] - reference_path_matrix[:, 1:N+1])

    # Vectorized dynamic constraints
    initial_eq_constraints = ca.reshape(X[:, 0] - start_state, -1, 1)
    next_states = f.map(N)(X[:, :-1], U)  # Compute all next states in one call
    subseq_eq_constraints = ca.reshape(X[:, 1:] - next_states, -1, 1)

    initial_control_eq_constraints = ca.reshape(U[:, 0] - start_control, -1, 1)

    accel_ineq_constraints =  ca.reshape((U[:, 1:] - U[:, :-1]).T, -1, 1) # TODO this can be optimized
    constraints = ca.vertcat(initial_eq_constraints,  # Initial state constraint
                             initial_control_eq_constraints, 
                             subseq_eq_constraints, # Dynamic constraints
                             accel_ineq_constraints)  

    lbg_list = [0] * (initial_eq_constraints.shape[0] + subseq_eq_constraints.shape[0] + initial_control_eq_constraints.shape[0])
    lbg_list += [-max_linear_accel] * (N - 1)
    lbg_list += [-max_angular_accel] * (N - 1)
    ubg_list = [0] * (initial_eq_constraints.shape[0] + subseq_eq_constraints.shape[0] + initial_control_eq_constraints.shape[0])
    ubg_list += [max_linear_accel] * (N - 1)
    ubg_list += [max_angular_accel] * (N - 1)

    # Define lower and upper bounds for U
    u_lb_states = ca.DM([0, -10, -3.14] * (N+1)) # TODO: 12/26/2024: hardcoded in robot's global ego view
    u_ub_states = ca.DM([20, 10, 3.14] * (N+1))
    u_lb = ca.DM([v_min, omega_min] * N)
    u_ub = ca.DM([v_max, omega_max] * N)
    
    # Create NLP dictionary
    nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),  # Flatten X and U
        'f': cost,
        'g': constraints}

    # Solver options
    opts = {
        "ipopt.print_level": 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-4,                # Adjust solver tolerance
        "ipopt.constr_viol_tol": 1e-3,   # Constraint violation tolerance
        "ipopt.acceptable_tol": 1e-3,     # Early stopping tolerance
        "print_time": False
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)

    # Initial guess
    X_init = ca.repmat(ca.DM(start_state), 1, N+1) # use start location as initial guess for faster convergence
    U_init = ca.repmat(ca.DM(start_control), 1, N)

    lbg = ca.DM(lbg_list)
    ubg = ca.DM(ubg_list)

    mid = time.time()
    # Solve the NLP
    solution = solver(
        x0=ca.vertcat(X_init.reshape((-1, 1)), U_init.reshape((-1, 1))),
        lbg=lbg, ubg=ubg,  
        lbx=ca.vertcat(u_lb_states, u_lb), 
        ubx=ca.vertcat(u_ub_states, u_ub)
    )

    # Extract optimal trajectories
    opt_X = solution["x"][:3*(N+1)].reshape((3, N+1))
    opt_U = solution["x"][3*(N+1):].reshape((2, N))
    
    end = time.time()
    return opt_X, opt_U

# verify xy seq
def verify_xy_seq(data):
    import matplotlib.pyplot as plt
    import cv2 
    import imageio

    grid_size = 32
    normalized_obstacle_seq = data["obs"]["normalized_obstacle_seq"]
    normalized_obstacle_seq = np.array(normalized_obstacle_seq)
    normalized_obstacle_seq = normalized_obstacle_seq * grid_size / 2
    fig, ax = plt.subplots()
    frames = []  
    for obs in normalized_obstacle_seq:
        ax.clear()
    
        for i in range(0, len(obs), 3):
            x = obs[i]
            y = obs[i+1]
            use = obs[i+2]
            if use == 0:
                continue
            circle = plt.Circle((y, x), 3, color='g')
            plt.gca().add_patch(circle)

        ax.grid(True)
        ax.axis([grid_size//2, -grid_size//2, -grid_size//2, grid_size//2])

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # Resize frame if necessary
        frame = cv2.resize(frame, (640, 380))
        # Convert RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frames.append(frame)
    # Save the frames as a gif
    save_name = "xy_seq"
    imageio.mimsave(f"{save_name}.gif", frames, fps=5, loop=0)
    print(f"Saving to {save_name}.gif")
    print(f"Done generating with name {save_name}")  
    plt.close()
    
def get_path(opt_u, start, dt):
    opt_u_np = np.array(opt_u)

    vw_path = []
    current_x = start[0]
    current_y = start[1]
    current_theta = start[2]
    for k in range(opt_u_np.shape[1]):
        v, w = opt_u_np[:, k]
        vw_path.append((current_x, current_y, current_theta, v, w))
        current_x, current_y, current_theta = forward_motion_rollout(v, w, current_x, current_y, current_theta, dt)
        
    last_v, last_w = opt_u_np[:, -1]
    vw_path.append((current_x, current_y, current_theta, last_v, last_w))    
        
    return vw_path