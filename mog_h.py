import argparse
import timeit
import subprocess
import IPython

import numpy as np
import math as m
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, Polygon, Point, LineString
from dm_control import suite
from dm_control import viewer
from dm_control.suite import pusher
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from PIL import Image

from resources import *

parser = argparse.ArgumentParser(description='Multi-Object-Grasping')
parser.add_argument('-l', '--obj_list', help='delimited list input', type=str)
parser.add_argument('-l1', '--obj_pts', help='delimited list input', type=str)
parser.add_argument('-l2', '--obj_widths', help='delimited list input', type=str)

args = parser.parse_args()
obj_names = [str(item) for item in args.obj_list.split(',')]
all_obj_num_pts = [int(item) for item in args.obj_pts.split(',')]
w_list = [float(item) for item in args.obj_widths.split(',')]

# obj_names = ['rr_1', 'y_1']
# all_obj_num_pts = [4,4]
# w_list = [0.015, 0.015]

num_objs = len(obj_names)

def initialize_physics_env():
    global env
    global physics
    global N_QPOS

    # Compose new object xml
    obj_names_str = ""
    count_objs = 0
    for obj_n in obj_names:
        obj_names_str = obj_names_str+obj_n
        if count_objs < len(obj_names)-1:
            obj_names_str = obj_names_str+','
        count_objs += 1

    subprocess.call(['python3', 'comp_objs.py', '--obj_list', obj_names_str])

    # Copy full system xml
    pusher_xml_path = suite.__path__[0]+"/pusher_clutter.xml"
    subprocess.call(["cp", "./mog_xmls/target.xml", pusher_xml_path])
    env = suite.load(domain_name="pusher", task_name="easy")
    physics = env.physics

    # Set physics simulation parameters
    physics.model.opt.timestep = SIM_TIMESTEP
    physics.model.opt.integrator = SIM_INTEGRATOR
    N_QPOS = physics.data.qpos[:].shape[0]
    x_0 = physics.get_state()
    env.physics.set_start_state(x_0[0:N_QPOS])

def set_state(x_0):
    # Reset physics and set initial state
    global env

    with env.physics.reset_context():
        env.physics.set_state(x_0)
        env.physics.step()

    return None

def visualize_grasp(x_0, cand_grasp):
    reset_viewer_policy_params(x_0, cand_grasp)
    viewer.launch(env, policy=policy)

def policy(timestep):
    global env
    return np.array([-1., 1.])

def reset_viewer_policy_params(x_0, grasp_cand):
    global env
    set_state(x_0)
    env.physics.model.opt.timestep = SIM_TIMESTEP
    env.physics.model.opt.integrator = SIM_INTEGRATOR
    env.physics.start_state = x_0[0:N_QPOS].copy()
    env.physics.set_start_state(x_0[0:N_QPOS])

def sim_mog_exp():
    global env
    global min_stable_distance

    print ('++++++++++++++++MOG exp. for ++++++++++++++: ', obj_names)

    base_data_path = './mog_data_{}/objs'.format(num_objs)
    for obj_ind in range(num_objs):
        base_data_path = base_data_path + '_' + obj_names[obj_ind]

    data_path = base_data_path + '/'

    gp_data_path = data_path+'grasp_planing/'
    sim_data_path = data_path+'sim_compare/'
    data_gen_path = data_path+'data_gen/'

    subprocess.call(['rm', '-rf', data_path])
    subprocess.call(['mkdir', '-p', data_path])
    subprocess.call(['rm', '-rf', data_gen_path])
    subprocess.call(['mkdir', '-p', data_gen_path])
    subprocess.call(['rm', '-rf', gp_data_path])
    subprocess.call(['mkdir', '-p', gp_data_path])
    subprocess.call(['rm', '-rf', sim_data_path])
    subprocess.call(['mkdir', '-p', sim_data_path])

    # Experiments in grasp planning
    all_bl1_times = np.zeros(NUM_SAMPLE_SCENES)
    all_bl2_times = np.zeros(NUM_SAMPLE_SCENES)
    all_bl3_times = np.zeros(NUM_SAMPLE_SCENES)
    all_gp_times = np.zeros(NUM_SAMPLE_SCENES)

    all_bl1_ss = []
    all_bl2_ss = []
    all_bl3_ss = []
    all_gp_ss = []

    all_bl1_used_samples = []
    all_bl2_used_samples = []
    all_bl3_used_samples = []
    all_gp_used_samples = []

    # Experiments comparing simulators
    all_mj_successes = []
    all_gp_successes = []
    all_gp_conditions = []
    all_initial_grasp_states = []

    grasp_labels = []
    grasp_center_pt_feats = []
    grasp_area_feats = []
    grasp_vert_feats = []
    grasp_delta_h = []
    grasp_curr_d = []
    grasp_stable_feats = []

    #+++++++++++++++++++++++Sample an initial state++++++++++++++++++++++++++
    for sample_scene in range(NUM_SAMPLE_SCENES):
        print ('Scene:', sample_scene)
        sim_scene_path = sim_data_path+'scene_{}/'.format(sample_scene)
        subprocess.call(['rm', '-rf', sim_scene_path])
        subprocess.call(['mkdir', '-p',sim_scene_path])
        print ('.')
        invalid_state = True
        MAX_TRIALS = 1000
        num_trials = 0

        desired_table_center = np.array([1.85, 0.0])

        while invalid_state and num_trials <= MAX_TRIALS:

            obj_quats = []
            obj_positions = []
            full_obj_state = []

            store_pos = 0
            eps_val = 1e-3
            for obj_ind in range(num_objs):
                obj_theta = np.random.uniform(low=-m.pi/10, high=m.pi/10)
                obj_quat = Quaternion(axis=[0,0,1], angle=obj_theta).elements

                if obj_ind == 0:
                    rel_val = eps_val + w_list[obj_ind]
                else:
                    rel_val = eps_val + w_list[obj_ind-1] + w_list[obj_ind]
                store_pos += rel_val

                obj_pos = desired_table_center - np.array([0, 0.04]) + np.array([0., store_pos])

                # Add variation in x
                y_var = np.random.uniform(low=-0.015, high=0.015)
                obj_pos[0] += y_var

                obj_quats.append(obj_quat)
                obj_positions.append(obj_pos)
                full_obj_state.append([obj_pos, obj_quat])

            # Update object states
            with env.physics.reset_context():
                for obj_ind in range(num_objs):
                    obj_name = obj_names[obj_ind]+"_joint"
                    obj_state =  full_obj_state[obj_ind].copy()
                    env.physics.named.data.qpos[obj_name][0:2] = obj_state[0].copy()
                    env.physics.named.data.qpos[obj_name][3:7] = obj_state[1].copy()

                env.physics.step()

            init_state = env.physics.get_state()
            invalid_state = check_valid_init_state()
            num_trials += 1

        if num_trials <= MAX_TRIALS:
            #print ('found sample!')
            pass

        if sample_scene == 0:
            # Compute this only once (only dependent on the objects not their state)
            min_stable_distance, _ = compute_min_stable_distance()
            #print ('Min stable dist is', min_stable_distance)

        #print ('Generating candidate grasps ....')
        cand_grasp_params, obj_state = gen_cand_grasps(init_state, plot_all_grasps=False, plot_path=sim_scene_path)

        cand_grasps = cand_grasp_params[0]

        std_indices = np.linspace(start=0, stop=len(cand_grasps), num=len(cand_grasps), endpoint=False)

        # # Baseline-1 algorithm (Randomize+simulate)
        # print ('Baseline 1 (Rand+Simulate)')
        rand_indices = np.random.permutation(len(cand_grasps))
        # bl1_start_time = timeit.default_timer()
        # bl1_grasp, bl1_results = baseline_algo(cand_grasps, full_obj_state, rand_indices)
        # bl1_time = timeit.default_timer() - bl1_start_time

        gp_params = [obj_state, std_indices]

        # Grasp planner (Rank+Filter+simulate)
        #print ('Grasp planner (ours)')
        gp_start_time = timeit.default_timer()
        gp_grasp, gp_results, sorted_indices, data_results = grasp_planner(cand_grasp_params, gp_params, full_obj_state, min_stable_distance, method="Area")
        gp_time = timeit.default_timer() - gp_start_time

        for data_ind in range(len(data_results)):
            grasp_labels.append(data_results[data_ind][0])
            grasp_center_pt_feats.append(data_results[data_ind][1])
            grasp_area_feats.append(data_results[data_ind][2])
            grasp_vert_feats.append(data_results[data_ind][3])
            grasp_delta_h.append(data_results[data_ind][4])
            grasp_curr_d.append(data_results[data_ind][5])
            grasp_stable_feats.append(data_results[data_ind][6])

        # # # Baseline-2 algorithm (Randomize+simulate)
        # print ('Baseline 2 (Rank+Simulate)')
        # bl2_start_time = timeit.default_timer()
        # bl2_grasp, bl2_results = baseline_algo(cand_grasps, full_obj_state, sorted_indices)
        # bl2_time = timeit.default_timer() - bl2_start_time

        # # Baseline 3 (Randomize+Filter+simulate)
        # print ('Baseline 3 (Rand+Filter+Simulate)')
        # bl3_start_time = timeit.default_timer()
        # bl3_grasp, bl3_results, _ = grasp_planner(cand_grasp_params, gp_params, full_obj_state, min_stable_distance, method="Random")
        # bl3_time = timeit.default_timer() - bl3_start_time
        #
        # bl1_success, bl1_used_samples, bl1_index = bl1_results
        # bl2_success, bl2_used_samples, bl2_index = bl2_results
        # bl3_success, bl3_used_samples, bl3_index = bl3_results
        # gp_success, gp_used_samples, gp_index = gp_results
        #
        # print ('Baseline-1 is {} in {} seconds with {} samples'.format(bl1_success, bl1_time, bl1_used_samples))
        # print ('Baseline-2 is {} in {} seconds with {} samples'.format(bl2_success, bl2_time, bl2_used_samples))
        # print ('Baseline-3 is {} in {} seconds with {} samples'.format(bl3_success, bl3_time, bl3_used_samples))
        # print ('Grasp planner is {} in {} seconds with {} samples'.format(gp_success, gp_time, gp_used_samples))
        # print ('\n')

        # # Store grasp states
        # save_state_fig(bas_state, gp_scene_path+'baseline_grasp_{}.png'.format(bas_index))
        # save_state_fig(gp_area_state, gp_scene_path+'gp_area_grasp_{}.png'.format(gp_area_index))

        # all_bl1_times[sample_scene] = bl1_time
        # all_bl2_times[sample_scene] = bl2_time
        # all_bl3_times[sample_scene] = bl3_time
        # all_gp_times[sample_scene] = gp_time
        #
        # all_bl1_ss.append(bl1_success)
        # all_bl2_ss.append(bl2_success)
        # all_bl3_ss.append(bl3_success)
        # all_gp_ss.append(gp_success)
        #
        # all_bl1_used_samples.append(bl1_used_samples)
        # all_bl2_used_samples.append(bl2_used_samples)
        # all_bl3_used_samples.append(bl3_used_samples)
        # all_gp_used_samples.append(gp_used_samples)
        #
        # # Compare Mujoco and Custom simulators
        # print ('Experiment to compare simulators')
        # mj_success, gp_success, gp_conditions, init_grasp_states = evaluate_grasps_exp(cand_grasp_params, gp_params, full_obj_state, sim_scene_path)
        #
        # # Initial analysis
        # sim_analysis(mj_success, gp_success)
        #
        # all_mj_successes.append(mj_success)
        # all_gp_successes.append(gp_success)
        # all_initial_grasp_states.append(init_grasp_states)
        # all_gp_conditions.append(gp_conditions)


    # Save grasp planning data
    np.save(data_gen_path+'grasp_labels', grasp_labels)
    np.save(data_gen_path+'grasp_center_pt_feats', grasp_center_pt_feats)
    np.save(data_gen_path+'grasp_area_feats', grasp_area_feats)
    np.save(data_gen_path+'grasp_vert_feats', grasp_vert_feats)
    np.save(data_gen_path+'grasp_delta_h', grasp_delta_h)
    np.save(data_gen_path+'grasp_curr_d', grasp_curr_d)
    np.save(data_gen_path+'grasp_stable_feats', grasp_stable_feats)
    # # Save grasp planning data
    # np.save(gp_data_path+'all_bl1_times', all_bl1_times)
    # np.save(gp_data_path+'all_bl2_times', all_bl2_times)
    # np.save(gp_data_path+'all_bl3_times', all_bl3_times)
    # np.save(gp_data_path+'all_gp_times', all_gp_times)
    #
    # np.save(gp_data_path+'all_bl1_ss', all_bl1_ss)
    # np.save(gp_data_path+'all_bl2_ss', all_bl2_ss)
    # np.save(gp_data_path+'all_bl3_ss', all_bl3_ss)
    # np.save(gp_data_path+'all_gp_ss', all_gp_ss)
    #
    # np.save(gp_data_path+'all_bl1_used_samples', all_bl1_used_samples)
    # np.save(gp_data_path+'all_bl2_used_samples', all_bl2_used_samples)
    # np.save(gp_data_path+'all_bl3_used_samples', all_bl3_used_samples)
    # np.save(gp_data_path+'all_gp_used_samples', all_gp_used_samples)
    #
    # np.save(sim_data_path+'all_mj_successes', all_mj_successes)
    # np.save(sim_data_path+'all_gp_successes', all_gp_successes)
    # np.save(sim_data_path+'all_initial_grasp_states', all_initial_grasp_states)
    # np.save(sim_data_path+'all_gp_conditions', all_gp_conditions)

    return None

def gen_dataset():
    global env
    global min_stable_distance

    print ('+++ Data collection for objects ++++ : ', obj_names)

    base_data_path = './mog_data_{}/objs'.format(num_objs)
    for obj_ind in range(num_objs):
        base_data_path = base_data_path + '_' + obj_names[obj_ind]

    rand_init_num = np.random.randint(0, 10000000)
    data_path = base_data_path + '/'
    sim_data_path = data_path+'sim_compare/'
    data_gen_path = data_path+'data_gen_{}/'.format(rand_init_num)

    subprocess.call(['rm', '-rf', data_gen_path])
    subprocess.call(['mkdir', '-p', data_gen_path])

    Xs_in = []
    Xs_feats_ours_in = []
    Xs_feats_baseline_in = []

    ys_in_ours = []
    grasp_labels_ours = []
    ys_in_ours_filter = []
    grasp_labels_filter = []
    ys_in_baseline = []
    grasp_labels_baseline = []
    all_stats = []

    #+++++++++++++++++++++++Sample an initial state++++++++++++++++++++++++++
    for sample_scene in range(NUM_SAMPLE_SCENES):
        # print ('Scene:', sample_scene)
        sim_scene_path = sim_data_path+'scene_{}/'.format(sample_scene)
        subprocess.call(['rm', '-rf', sim_scene_path])
        subprocess.call(['mkdir', '-p',sim_scene_path])
        # print ('.')
        invalid_state = True
        MAX_TRIALS = 1000
        num_trials = 0

        desired_table_center = np.array([1.85, 0.0])

        while invalid_state and num_trials <= MAX_TRIALS:

            obj_quats = []
            obj_positions = []
            full_obj_state = []

            store_pos = 0
            eps_val = 1e-3
            for obj_ind in range(num_objs):
                obj_theta = np.random.uniform(low=-m.pi/10, high=m.pi/10)
                obj_quat = Quaternion(axis=[0,0,1], angle=obj_theta).elements

                if obj_ind == 0:
                    rel_val = eps_val + w_list[obj_ind]
                else:
                    rel_val = eps_val + w_list[obj_ind-1] + w_list[obj_ind]
                store_pos += rel_val

                obj_pos = desired_table_center - np.array([0, 0.04]) + np.array([0., store_pos])

                # Add variation in x
                y_var = np.random.uniform(low=-0.015, high=0.015)
                obj_pos[0] += y_var

                obj_quats.append(obj_quat)
                obj_positions.append(obj_pos)
                full_obj_state.append([obj_pos, obj_quat])

            # Update object states
            with env.physics.reset_context():
                for obj_ind in range(num_objs):
                    obj_name = obj_names[obj_ind]+"_joint"
                    obj_state =  full_obj_state[obj_ind].copy()
                    env.physics.named.data.qpos[obj_name][0:2] = obj_state[0].copy()
                    env.physics.named.data.qpos[obj_name][3:7] = obj_state[1].copy()

                env.physics.step()

            init_state = env.physics.get_state()
            invalid_state = check_valid_init_state()
            num_trials += 1

        if num_trials <= MAX_TRIALS:
            #print ('found sample!')
            pass

        if sample_scene == 0:
            # Compute this only once (only dependent on the objects not their state)
            min_stable_distance, _ = compute_min_stable_distance()
            #print ('Min stable dist is', min_stable_distance)

        #print ('Generating candidate grasps ....')
        cand_grasp_params, obj_state = gen_cand_grasps(init_state, plot_all_grasps=False, plot_path=sim_scene_path)

        cand_grasps = cand_grasp_params[0]

        std_indices = np.linspace(start=0, stop=len(cand_grasps), num=len(cand_grasps), endpoint=False)

        rand_indices = np.random.permutation(len(cand_grasps))

        gp_params = [obj_state, std_indices]

        # gp_start_time = timeit.default_timer()
        # gp_grasp, gp_results, sorted_indices, data_results = grasp_planner(cand_grasp_params, gp_params, full_obj_state, min_stable_distance, method="Area")
        # gp_time = timeit.default_timer() - gp_start_time

        data_ours, data_ours_fil, data_baseline, stats, X_in, X_feats_ours, X_feats_baseline = gen_data_learn( cand_grasp_params,
                                                                                gp_params, full_obj_state, min_stable_distance)

        y_in_ours, grasp_label_ours = data_ours
        y_in_ours_filters, grasp_label_filters = data_ours_fil
        y_in_baseline, grasp_label_baseline = data_baseline

        Xs_in.append(X_in)

        Xs_feats_ours_in.append(X_feats_ours)
        Xs_feats_baseline_in.append(X_feats_baseline)

        ys_in_ours.append(y_in_ours)
        grasp_labels_ours.append(grasp_label_ours)

        ys_in_ours_filter.append(y_in_ours_filters)
        grasp_labels_filter.append(grasp_label_filters)

        ys_in_baseline.append(y_in_baseline)
        grasp_labels_baseline.append(grasp_label_baseline)
        all_stats.append(stats)

    # Save grasp planning data
    # One grasp per scene ! -

    np.save(data_gen_path+'Xs_in', Xs_in)
    np.save(data_gen_path+'Xs_feats_ours_in', Xs_feats_ours_in)
    np.save(data_gen_path+'Xs_feats_baseline_in', Xs_feats_baseline_in)

    # Our approach main data
    np.save(data_gen_path+'ys_in_ours', ys_in_ours)
    np.save(data_gen_path+'grasp_labels_ours', grasp_labels_ours)

    # Our approach filter data
    np.save(data_gen_path+'ys_in_ours_filter', ys_in_ours_filter)
    np.save(data_gen_path+'grasp_labels_filter', grasp_labels_filter)

    # Baseline main data
    np.save(data_gen_path+'ys_in_baseline', ys_in_baseline)
    np.save(data_gen_path+'grasp_labels_baseline', grasp_labels_baseline)
    np.save(data_gen_path+'all_stats', all_stats)

    #IPython.embed()
    # print (ys_in_ours_filter)
    # print (ys_in_baseline)
    return None

def compute_min_stable_distance():

    obj_pts, obj_lines, obj_edge_lines = get_props()

    # For each object, compute stability type and corresponding distance
    all_min_dists = []
    for obj_ind in range(num_objs):
        _, dists = compute_stable_configs_and_dists(obj_pts[obj_ind], obj_lines[obj_ind], obj_edge_lines[obj_ind])
        min_dist = np.min(dists)
        all_min_dists.append(min_dist)

    min_mog_dist = np.sum(np.array(all_min_dists))
    indiv_min = np.min(all_min_dists)

    return min_mog_dist, indiv_min

def compute_stable_configs_and_dists(obj_pts, obj_lines, obj_edge_lines):

    stable_configs = []
    stable_dists = []

    # Two parallel lines belonging to the object.
    num_lines = len(obj_lines)
    for line_a_ind in range(num_lines):
        for line_b_ind in range(line_a_ind+1, num_lines):
            line_a = obj_lines[line_a_ind]
            line_b = obj_lines[line_b_ind]
            parallel_lines = check_parallel_lines(line_a, line_b)
            if parallel_lines:
                stable_config_type = 'l_l'
                stable_config_dist = LineString(line_a).distance(LineString(line_b))
                stable_configs.append(stable_config_type)
                stable_dists.append(stable_config_dist)

    num_edge_points = len(obj_pts)

    # Edge point and corresponding pependicular line.
    for edge_pt_ind in range(num_edge_points):
        for line_ind in range(num_lines):
            # Use the full line instead of the intersection line only.
            line = obj_lines[line_ind]
            edge_pt = obj_pts[edge_pt_ind]
            edge_lines = obj_edge_lines[edge_pt_ind]
            line_edge_point = check_line_edge_point_pepend(line, edge_pt, edge_lines)
            if line_edge_point:
                stable_config_type = 'l_p'
                stable_config_dist = LineString(line).distance(Point(edge_pt))
                stable_configs.append(stable_config_type)
                stable_dists.append(stable_config_dist)

    # Two edge points that are pependicularly connected.
    for edge_pt_ind_1 in range(num_edge_points):
        for edge_pt_ind_2 in range(edge_pt_ind_1+1, num_edge_points):
            edge_pt_1 = obj_pts[edge_pt_ind_1]
            edge_pt_2 = obj_pts[edge_pt_ind_2]
            edge_pt_1_lines = obj_edge_lines[edge_pt_ind_1]
            edge_pt_2_lines = obj_edge_lines[edge_pt_ind_2]
            pts = [edge_pt_1, edge_pt_2]
            lines = [edge_pt_1_lines, edge_pt_2_lines]
            edge_pepend = check_edge_pepend_connection(pts, lines)
            if edge_pepend:
                stable_config_type = 'p_p'
                stable_config_dist = np.linalg.norm(np.array(edge_pt_2) - np.array(edge_pt_1))
                stable_configs.append(stable_config_type)
                stable_dists.append(stable_config_dist)

    return stable_configs, stable_dists

def get_props():

    obj_state, pt_list = get_object_state()
    all_obj_pts = []
    all_obj_lines = []
    all_obj_edge_lines = []

    for obj_ind in range(num_objs):
        hull_object = ConvexHull(obj_state[obj_ind])
        hull_path = Path(hull_object.points[hull_object.vertices])
        obj_polygon = Polygon(hull_path.to_polygons()[0])
        obj_polygon_coords = wrap_points(obj_polygon.boundary.xy)[0:-1]
        obj_lines = generate_lines_from_pts(obj_polygon_coords)
        obj_edge_pts = obj_polygon_coords
        obj_edge_lines = []
        for edge_pt in obj_edge_pts:
            edge_lines = extract_edge_lines(edge_pt, obj_lines)
            obj_edge_lines.append(edge_lines)

        all_obj_pts.append(obj_edge_pts)
        all_obj_lines.append(obj_lines)
        all_obj_edge_lines.append(obj_edge_lines)

    return all_obj_pts, all_obj_lines, all_obj_edge_lines

def extract_edge_lines(edge_pt, obj_lines):

    edge_lines = []
    SMALL_EPS = 1e-6
    for line in obj_lines:
        p_l_dist = LineString(line).distance(Point(edge_pt))
        if p_l_dist < SMALL_EPS:
            edge_lines.append(line)

    return edge_lines

def check_valid_init_state():

    # No obj_obj collisions and no plate_obj collisions.
    obj_obj_collisions = False
    obj_state, pt_list = get_object_state()
    hull = ConvexHull(np.array(pt_list))

    obj_polygons = []
    obj_hulls = []
    for obj_ind in range(num_objs):
        hull_object = ConvexHull(obj_state[obj_ind])
        hull_path = Path(hull_object.points[hull_object.vertices])
        obj_polygon = Polygon(hull_path.to_polygons()[0])
        obj_polygons.append(obj_polygon)
        obj_hulls.append(hull_object)

    obj_obj_collisions = False
    for obj_ind_1 in range(num_objs):
        for obj_ind_2 in range(obj_ind_1+1, num_objs):
            if obj_polygons[obj_ind_1].intersects(obj_polygons[obj_ind_2]):
                obj_obj_collisions = True
                break

    return obj_obj_collisions

def simulate_grasp_gp(sim_grasp_input, print_flag=False):

    """ Check necessary conditions for multi-object grasping."""

    swept_polygon = sim_grasp_input[0]
    obj_pts = sim_grasp_input[1]
    plate_polygons = sim_grasp_input[2]
    object_int_polygons = sim_grasp_input[3]
    object_polygons = sim_grasp_input[4]

    # Intersection area condition
    int_area_cond = int_area_condition_check(object_int_polygons)

    # Diameter function check
    diameter_cond, delta_h, curr_d = diameter_condition_check(plate_polygons, object_int_polygons)

    if int_area_cond == False or diameter_cond == False:
        grasp_success = 'False'
    else:
        grasp_success = 'Maybe'

    failure_conds = [int_area_cond, diameter_cond]

    return grasp_success, failure_conds, delta_h, curr_d

def diameter_condition_check(plate_polygons, object_int_polygons):

    current_d = compute_current_distance_func(plate_polygons, object_int_polygons)

    if min_stable_distance <= current_d:
        diameter_cond = True
    else:
        diameter_cond = False

    delta_h = current_d - min_stable_distance

    return diameter_cond, delta_h, current_d

def compute_current_distance_func(plate_polygons, object_int_polygons):

    l_plate_polygon = plate_polygons[0]
    r_plate_polygon = plate_polygons[1]

    lp_distances = []
    for obj_ind in range(num_objs):
        lp_dist = l_plate_polygon.distance(object_int_polygons[obj_ind])
        lp_distances.append(lp_dist)

    rp_distances = []
    for obj_ind in range(num_objs):
        rp_dist = r_plate_polygon.distance(object_int_polygons[obj_ind])
        rp_distances.append(rp_dist)

    min_dist_to_left_plate = np.min(lp_distances)
    min_dist_to_right_plate = np.min(rp_distances)

    # Distance between left plate polygon and right plate polygon
    plate_plate_init_dist = l_plate_polygon.distance(r_plate_polygon)

    total_free_space = min_dist_to_left_plate + min_dist_to_right_plate
    dist_func_ini = plate_plate_init_dist - total_free_space

    return dist_func_ini

def grasp_planner(cand_grasp_params, gp_params, obj_params, h_fmin, method='Area'):

    """Sorts all candidate grasps using 'Method', filters inadmissible grasps
        and simulates potential grasps with Mujoco until a grasp is found,
        or all grasps are tested.
    """

    plan_grasp = True
    #print ('hfmin is', h_fmin)
    #print ('Max is', G_STROKE - G_WIDTH*2 )
    # Check if the grasp can even happen:
    if h_fmin > G_STROKE - G_WIDTH*2:
        # Gripper can't hold all objects in a grasp
        plan_grasp = False

    if plan_grasp:
        # General grasp params
        cand_grasps = cand_grasp_params[0]
        cand_grasp_areas = cand_grasp_params[1]

        # Params to check failure
        cand_grasp_swept_polygon = cand_grasp_params[2]
        cand_grasp_plate_polygons = cand_grasp_params[3]
        cand_grasp_obj_int_polygons = cand_grasp_params[4]
        cand_grasp_grippers_hull_path = cand_grasp_params[-1]

        sorted_cand_grasps, sorted_indices = sort_all_grasps(cand_grasps, cand_grasp_areas, gp_params, sort_method=method)

        # Compute centroid of objects in current state
        obj_state = gp_params[0].copy()

        obj_centroids = []
        obj_polygons = []
        all_obj_pts = []
        for obj_ind in range(num_objs):
            obj_pts = obj_state[obj_ind].copy()
            obj_centroids.append(np.mean(obj_pts, axis=0))

            hull_object = ConvexHull(obj_pts)
            hull_path = Path(hull_object.points[hull_object.vertices])
            obj_polygon = Polygon(hull_path.to_polygons()[0])
            obj_polygons.append(obj_polygon)
            all_obj_pts.append(obj_pts)

        count_used_samples = 0
        all_data_results = []
        for grasp_index in range(len(cand_grasps)):
            grasp_success = 'False'
            #print ('Current grasp:', int(sorted_indices[grasp_index]))

            count_used_samples += 1

            cand_grasp = sorted_cand_grasps[grasp_index]
            swept_polygon = cand_grasp_swept_polygon[int(sorted_indices[grasp_index])]
            plate_polygons = cand_grasp_plate_polygons[int(sorted_indices[grasp_index])]
            obj_int_polygons = cand_grasp_obj_int_polygons[int(sorted_indices[grasp_index])]
            grippers_hull_path = cand_grasp_grippers_hull_path[int(sorted_indices[grasp_index])]

            sim_grasp_input = []
            sim_grasp_input.append(swept_polygon)
            sim_grasp_input.append(all_obj_pts)
            sim_grasp_input.append(plate_polygons)
            sim_grasp_input.append(obj_int_polygons)
            sim_grasp_input.append(obj_polygons)

            grasp_success_gp, failure_conds, delta_h, curr_d = simulate_grasp_gp(sim_grasp_input)

            #print ('Success is {} with {} conditions @ index {}'.format(grasp_success_gp, failure_conds, int(sorted_indices[grasp_index])))

            if grasp_success_gp == 'Maybe':

                init_grasp_state, final_grasp_state, grasp_success_mj, grasp_X = simulate_full_grasp(obj_params, cand_grasp)
                visualize_grasp(init_grasp_state, cand_grasp)

                g1_pos, g2_pos, g1_quat, g2_quat = get_gripper_params(cand_grasp)
                center_pt_mean, center_pt_std, center_pt_dists = get_center_pt_feature(obj_centroids, [g1_pos, g2_pos])
                area_feature_mean, area_feature_std = get_area_feature(obj_int_polygons)
                vert_mean, vert_std = get_vertices_feature(all_obj_pts)

                # Stable configs
                # Lines describing the swept polygon
                sp_vert = grippers_hull_path.vertices
                swept_polygon_lines = generate_lines_from_pts(sp_vert)

                all_obj_intersect_coords = []
                all_obj_intersect_lines = []
                all_obj_polygon_coords = []

                for obj_ind in range(num_objs):
                    obj_int_polygon = obj_int_polygons[obj_ind]
                    obj_polygon = obj_polygons[obj_ind]
                    obj_polygon_coords = wrap_points(obj_polygon.boundary.xy)[0:-1]
                    obj_intersect_coords = wrap_points(obj_int_polygon.boundary.xy)[0:-1]
                    obj_intersect_lines = generate_lines_from_pts(obj_intersect_coords)
                    all_obj_intersect_coords.append(obj_intersect_coords)
                    all_obj_intersect_lines.append(obj_intersect_lines)
                    all_obj_polygon_coords.append(obj_polygon_coords)

                obj_props_input = []
                obj_props_input.append(swept_polygon_lines)
                obj_props_input.append(all_obj_intersect_lines)
                obj_props_input.append(all_obj_intersect_coords)
                obj_props_input.append(all_obj_polygon_coords)
                stable_config_mean, stable_config_std = get_stable_configs_feature(obj_props_input)

                #visualize_grasp(init_grasp_state, cand_grasp)
                # print ('\n')
                # #print ('Grasp: {} , Midline feature: {}'.format(grasp_success_mj, center_pt_feature))
                #print ('Grasp:', grasp_success_mj)
                #print ('Stable feature', [stable_config_mean, stable_config_std])

                # print ('Area: {} +/- {}'.format(area_feature_mean*1e3, area_feature_std*1e3))
                # print ('Centerline: {} +/- {}'.format(center_pt_mean*1e3, center_pt_std*1e3))
                # print ('Vertices: {} +/- {}'.format(vert_mean, vert_std))

                if grasp_success_mj == 'True':
                    grasp_success = 'True'

                else:
                    grasp_success = 'False'

                data_results = [grasp_success, [center_pt_mean, center_pt_std],
                [area_feature_mean, area_feature_std], [vert_mean, vert_std], delta_h, curr_d,
                [stable_config_mean, stable_config_std]]

                all_data_results.append(data_results)

            else:
                dum_cst = 100
                center_pt_mean = dum_cst
                center_pt_std = dum_cst
                center_pt_dists = []
                area_feature_mean = dum_cst
                area_feature_std = dum_cst
                vert_mean = dum_cst
                vert_std = dum_cst

            # if grasp_success == 'True':
            #     break

            results = [grasp_success, count_used_samples, int(sorted_indices[grasp_index]),
                        center_pt_mean, center_pt_dists]


    else:
        cand_grasp = []
        results = ['False', 0, 0]
        sorted_indices = []

    return cand_grasp, results, sorted_indices, all_data_results

def gen_data_learn(cand_grasp_params, gp_params, obj_params, h_fmin):

    # General grasp params
    cand_grasps = cand_grasp_params[0]
    cand_grasp_areas = cand_grasp_params[1]

    # Params to check failure
    cand_grasp_swept_polygon = cand_grasp_params[2]
    cand_grasp_plate_polygons = cand_grasp_params[3]
    cand_grasp_obj_int_polygons = cand_grasp_params[4]
    cand_grasp_grippers_hull_path = cand_grasp_params[-1]

    # Compute centroid of objects in current state
    obj_state = gp_params[0].copy()

    obj_centroids = []
    obj_polygons = []
    all_obj_pts = []
    for obj_ind in range(num_objs):
        obj_pts = obj_state[obj_ind].copy()
        obj_centroids.append(np.mean(obj_pts, axis=0))

        hull_object = ConvexHull(obj_pts)
        hull_path = Path(hull_object.points[hull_object.vertices])
        obj_polygon = Polygon(hull_path.to_polygons()[0])
        obj_polygons.append(obj_polygon)
        all_obj_pts.append(obj_pts)


    filters_labels = []
    filters_grasps = []
    promising_grasps = []
    grasps_delta_h = []
    grasps_curr_d = []
    grasps_grippers_hull_path = []
    for grasp_index in range(len(cand_grasps)):
        cand_grasp = cand_grasps[grasp_index]
        swept_polygon = cand_grasp_swept_polygon[grasp_index]
        plate_polygons = cand_grasp_plate_polygons[grasp_index]
        obj_int_polygons = cand_grasp_obj_int_polygons[grasp_index]
        grippers_hull_path = cand_grasp_grippers_hull_path[grasp_index]

        sim_grasp_input = []
        sim_grasp_input.append(swept_polygon)
        sim_grasp_input.append(all_obj_pts)
        sim_grasp_input.append(plate_polygons)
        sim_grasp_input.append(obj_int_polygons)
        sim_grasp_input.append(obj_polygons)

        grasp_success_gp, failure_conds, delta_h, curr_d = simulate_grasp_gp(sim_grasp_input)

        grasps_curr_d.append(curr_d)
        grasps_delta_h.append(delta_h)
        grasps_grippers_hull_path.append(grippers_hull_path)

        if grasp_success_gp == 'Maybe':
            # Promising grasp
            promising_grasps.append(grasp_index)
        else:
            # Use this grasp to learn the filtering
            filters_labels.append(0)
            filters_grasps.append(cand_grasp)

    # Pick a promising grasp at random
    if len(promising_grasps) != 0:
        rand_num = np.random.randint(0, len(promising_grasps))
        rand_prom_grasp_index = promising_grasps[rand_num]

    else:
        rand_num = np.random.randint(0, len(cand_grasps))
        rand_prom_grasp_index = rand_num

    cand_grasp_prom = cand_grasps[rand_prom_grasp_index]
    i_state_prom, _, label_prom, _ = simulate_full_grasp(obj_params, cand_grasp_prom)
    visualize_grasp(i_state_prom, cand_grasp_prom)

    # Pick a grasp at random
    rand_grasp_index = np.random.randint(0, len(cand_grasps))
    cand_grasp_baseline = cand_grasps[rand_grasp_index]
    i_state_baseline, _, label_baseline, _ = simulate_full_grasp(obj_params, cand_grasp_baseline)
    # visualize_grasp(i_state_baseline, cand_grasp_baseline)

    data_ours  = [cand_grasp_prom, label_prom]
    data_baseline = [cand_grasp_baseline, label_baseline]
    data_ours_fil = [filters_grasps, filters_labels]
    stats = [len(promising_grasps), len(cand_grasps)]

    X_in = construct_xin(all_obj_pts)

    obj_int_polygons_prom = cand_grasp_obj_int_polygons[rand_prom_grasp_index]
    obj_int_polygons_bas = cand_grasp_obj_int_polygons[rand_grasp_index]

    prom_delta_h = grasps_delta_h[rand_prom_grasp_index]
    bas_delta_h = grasps_delta_h[rand_grasp_index]

    prom_curr_d = grasps_curr_d[rand_prom_grasp_index]
    bas_curr_d = grasps_curr_d[rand_grasp_index]

    grippers_hull_path_prom = grasps_grippers_hull_path[rand_prom_grasp_index]
    grippers_hull_path_bas = grasps_grippers_hull_path[rand_grasp_index]

    X_features_in_ours = construct_features(cand_grasp_prom, obj_centroids,
            obj_int_polygons_prom, obj_polygons, all_obj_pts, prom_curr_d, prom_delta_h, grippers_hull_path_prom)
    X_features_in_baseline = construct_features(cand_grasp_baseline, obj_centroids,
            obj_int_polygons_bas, obj_polygons, all_obj_pts, bas_curr_d, bas_delta_h, grippers_hull_path_bas)

    # print ('Stats', stats)
    return data_ours, data_ours_fil, data_baseline, stats, X_in, X_features_in_ours, X_features_in_baseline

def construct_features(cand_grasp, obj_centroids, obj_int_polygons, obj_polygons, all_obj_pts, curr_d, delta_h, grippers_hull_path):

    g1_pos, g2_pos, g1_quat, g2_quat = get_gripper_params(cand_grasp)
    center_pt_mean, center_pt_std, center_pt_dists = get_center_pt_feature(obj_centroids, [g1_pos, g2_pos])
    area_feature_mean, area_feature_std = get_area_feature(obj_int_polygons)
    vert_mean, vert_std = get_vertices_feature(all_obj_pts)

    # Stable configs
    # Lines describing the swept polygon
    sp_vert = grippers_hull_path.vertices
    swept_polygon_lines = generate_lines_from_pts(sp_vert)

    all_obj_intersect_coords = []
    all_obj_intersect_lines = []
    all_obj_polygon_coords = []

    for obj_ind in range(num_objs):
        obj_int_polygon = obj_int_polygons[obj_ind]
        obj_polygon = obj_polygons[obj_ind]
        obj_polygon_coords = wrap_points(obj_polygon.boundary.xy)[0:-1]
        try:
            obj_intersect_coords = wrap_points(obj_int_polygon.boundary.xy)[0:-1]
        except:
            obj_intersect_coords = []

        try:
            obj_intersect_lines = generate_lines_from_pts(obj_intersect_coords)
        except:
            obj_intersect_lines = []

        all_obj_intersect_coords.append(obj_intersect_coords)
        all_obj_intersect_lines.append(obj_intersect_lines)
        all_obj_polygon_coords.append(obj_polygon_coords)

    obj_props_input = []
    obj_props_input.append(swept_polygon_lines)
    obj_props_input.append(all_obj_intersect_lines)
    obj_props_input.append(all_obj_intersect_coords)
    obj_props_input.append(all_obj_polygon_coords)
    stable_config_mean, stable_config_std = get_stable_configs_feature(obj_props_input)

    data_results = [[center_pt_mean, center_pt_std],
    [area_feature_mean, area_feature_std], [vert_mean, vert_std], delta_h, curr_d,
    [stable_config_mean, stable_config_std]]

    return data_results

def get_vertices_feature(all_obj_pts):

    all_verts = []
    for obj_pts in all_obj_pts:
        all_verts.append(len(obj_pts))

    vert_mean = np.mean(all_verts)
    vert_std = np.std(all_verts)

    return vert_mean, vert_std

def construct_xin(all_obj_pts):

    # At most 4 objects each with at most 6 vertices
    x_in = np.zeros((24, 2))

    count_objs = 0
    for obj_pts in all_obj_pts:
        count_verts = 6*count_objs
        for obj_pt in obj_pts:
            x_in[count_verts] = obj_pt.copy()
            count_verts += 1
        count_objs += 1

    # print (x_in)
    return x_in

def get_area_feature(obj_int_polygons):

    all_areas = []
    for obj_ind in range(num_objs):
        all_areas.append(obj_int_polygons[obj_ind].area)

    area_feature_mean = np.mean(all_areas)
    area_feature_std = np.std(all_areas)

    return area_feature_mean, area_feature_std

def get_center_pt_feature(obj_centroids, grasp_midline):

    center_pt_dists = []
    for obj_ind in range(num_objs):
        obj_centroid = obj_centroids[obj_ind]
        center_pt_dist = compute_center_pt_dists(obj_centroid, grasp_midline)
        center_pt_dists.append(center_pt_dist)

    center_pt_mean = np.mean(center_pt_dists)
    center_pt_std = np.std(center_pt_dists)

    return center_pt_mean, center_pt_std, center_pt_dists

def compute_center_pt_dists(centroid_pt, line_pts):

    center_pt_dist = LineString(line_pts).distance(Point(centroid_pt))

    return center_pt_dist

def extract_full_lines(obj_inter_lines, obj_full_lines):

    obj_int_full_lines = []

    SMALL_EPS =  1e-6
    for line_ind in range(len(obj_inter_lines)):
        line_of_interest = obj_inter_lines[line_ind]
        # Find the corresponding object full line.
        # It is parallel and intersects
        for d_line in obj_full_lines:
            if check_parallel_lines(line_of_interest, d_line):
                if LineString(d_line).distance(LineString(line_of_interest)) < SMALL_EPS:
                    obj_int_full_lines.append(d_line)

    # print ('Num of orig. lines', len(obj_inter_lines))
    # print ('NUm of full lines', len(obj_int_full_lines))
    # print ('\n')

    if len(obj_inter_lines) != len(obj_int_full_lines):
        print ('Sound the alarm!!!!==========================')
        #IPython.embed()

    # obj_inter_lines_x = []
    # obj_inter_lines_y = []
    # for line_int in obj_inter_lines:
    #     obj_inter_lines_x.append(line_int[0][0])
    #     obj_inter_lines_x.append(line_int[1][0])
    #     obj_inter_lines_y.append(line_int[0][1])
    #     obj_inter_lines_y.append(line_int[1][1])
    #
    # obj_full_line_x = []
    # obj_full_line_y = []
    # for line_full in obj_full_lines:
    #     obj_full_line_x.append(line_full[0][0])
    #     obj_full_line_x.append(line_full[1][0])
    #     obj_full_line_y.append(line_full[0][1])
    #     obj_full_line_y.append(line_full[1][1])
    #
    # plt.cla()
    # plt.plot(obj_full_line_x, obj_full_line_y, 'b')
    # plt.plot(obj_inter_lines_x, obj_inter_lines_y, 'r')
    # plt.savefig('Full_line_test.png')

    return obj_int_full_lines

def compute_obj_props(obj_props_input):

    swept_polygon_lines = obj_props_input[0]
    all_obj_intersect_lines = obj_props_input[1]
    all_obj_intersect_coords = obj_props_input[2]
    all_obj_polygon_coords = obj_props_input[3]

    all_obj_own_lines = []
    all_obj_int_pts = []
    all_obj_edge_lines = []
    all_obj_all_lines = []

    for obj_ind in range(num_objs):
        obj_intersect_lines = all_obj_intersect_lines[obj_ind]
        obj_intersect_coords = all_obj_intersect_coords[obj_ind]
        obj_polygon_coords = all_obj_polygon_coords[obj_ind]

        obj_own_lines, obj_int_pts, obj_edge_lines = filter_lines(swept_polygon_lines,
                                    obj_intersect_lines, obj_intersect_coords)

        obj_all_lines = generate_lines_from_pts(obj_polygon_coords)

        all_obj_own_lines.append(obj_own_lines)
        all_obj_int_pts.append(obj_int_pts)
        all_obj_edge_lines.append(obj_edge_lines)
        all_obj_all_lines.append(obj_all_lines)

    all_obj_full_edge_lines = []
    all_obj_full_own_lines = []
    for obj_ind in range(num_objs):
        obj_own_lines = all_obj_own_lines[obj_ind]
        obj_all_lines = all_obj_all_lines[obj_ind]
        obj_edge_lines = all_obj_edge_lines[obj_ind]

        obj_full_own_lines = extract_full_lines(obj_own_lines, obj_all_lines)
        all_obj_full_own_lines.append(obj_full_own_lines)
        obj_full_edge_lines = []
        for edge_pt in range(len(obj_edge_lines)):
            full_edge_lines = extract_full_lines(obj_edge_lines[edge_pt], obj_all_lines)
            obj_full_edge_lines.append(full_edge_lines)

        all_obj_full_edge_lines.append(obj_full_edge_lines)

    obj_props = [all_obj_own_lines, all_obj_int_pts, all_obj_edge_lines,
                all_obj_full_own_lines, all_obj_full_edge_lines]

    return obj_props

def filter_lines(swept_polygon_lines, obj_lines, obj_vertices):

    # Hack
    # Lines of interest are only the long sides of the swept polygon
    # (not the actual gripper line).

    # Only 2 sp lines with the highest magnitude are of interest
    # Assuming gripper width is less than the stroke.

    line_mags = []
    for swp in swept_polygon_lines:
        line_vec = np.array(np.array(swp[1]) - np.array(swp[0]))
        line_mag = np.linalg.norm(line_vec)
        line_mags.append(line_mag)

    max_mag = np.max(np.array(line_mags))
    SMALL_EPS = 1e-6
    sp_line_indexes = np.where(abs(np.array(line_mags) - max_mag) < SMALL_EPS)[0]

    filtered_lines = []
    discarded_lines = []
    #full_filtered_lines = []

    obj_line_index = 0
    for obj_line in obj_lines:
        contains  = False
        #obj_full_line = full_obj_lines[obj_line_index]
        for sp_line_index in sp_line_indexes:
            # Check if sp_line contains obj_line
            sp_line  = swept_polygon_lines[sp_line_index]
            if check_parallel_lines(sp_line, obj_line):
                if abs(LineString(sp_line).distance(LineString(obj_line))) < SMALL_EPS:
                    # Line belongs to the swept polygon
                    #print ('Contains case found')
                    contains = True
        if contains:
            discarded_lines.append(obj_line)
        else:
            filtered_lines.append(obj_line)
            #full_filtered_lines.append(obj_full_line)

        obj_line_index += 1

    # sp_x = [sp_line[0][0] , sp_line[1][0]]
    # sp_y = [sp_line[0][1], sp_line[1][1]]
    #
    # obj_x = [obj_line[0][0] , obj_line[1][0]]
    # obj_y = [obj_line[0][1], obj_line[1][1]]
    #
    # plt.cla()
    # plt.plot(sp_x, sp_y, 'b')
    # plt.plot(obj_x, obj_y, 'r')
    # plt.savefig('test.png')

    obj_int_pts = []
    for obj_vert in obj_vertices:
        count_d = 0
        for d_line in discarded_lines:
            if abs(LineString(d_line).distance(Point(obj_vert))) < SMALL_EPS:
                # Point lies on the discarded line
                count_d += 1
        if count_d == 0:
            # Interior points (not lying on the discarded line)
            obj_int_pts.append(obj_vert)

    # Edge lines belonging to each interior obj point
    obj_edge_lines = []
    #obj_full_edge_lines = []
    for int_obj_pt in obj_int_pts:
        pt_edge_lines = []
        #pt_full_edge_lines = [] # The full line (not just the part in the swept area)
        f_line_index = 0
        for f_line in filtered_lines:
            #full_f_line = full_filtered_lines[f_line_index]
            if abs(LineString(f_line).distance(Point(int_obj_pt))) < SMALL_EPS:
                pt_edge_lines.append(f_line)
                #pt_full_edge_lines.append(full_f_line)
        f_line_index += 1
        obj_edge_lines.append(pt_edge_lines)
        #obj_full_edge_lines.append(pt_full_edge_lines)

    return filtered_lines, obj_int_pts, obj_edge_lines

def compute_num_stable_configs(stable_config_params, debug_flag=False):

    # Count number of stable configs in an intersection polygon
    obj_own_lines = stable_config_params[0]
    obj_int_pts = stable_config_params[1]
    obj_edge_lines = stable_config_params[2]
    obj_full_own_lines = stable_config_params[3]
    obj_full_edge_lines = stable_config_params[4]

    # Two parallel lines belonging to the object.
    parallel_lines = False
    num_lines = len(obj_own_lines)
    the_parallel_lines = []
    count_parallel_lines = 0
    for line_a_ind in range(num_lines):
        for line_b_ind in range(line_a_ind+1, num_lines):
            line_a = obj_own_lines[line_a_ind]
            line_b = obj_own_lines[line_b_ind]
            parallel_lines = check_parallel_lines(line_a, line_b)
            if parallel_lines:
                count_parallel_lines += 1
                the_parallel_lines.append([line_a, line_b])

    num_edge_points = len(obj_int_pts)

    # Edge point and corresponding pependicular line.
    line_edge_point = False
    the_line_edge_points = []
    count_line_edge_points = 0
    for edge_pt_ind in range(num_edge_points):
        for line_ind in range(len(obj_full_own_lines)):
            #line = obj_own_lines[line_ind]
            # Use the full line instead of the intersection line only.
            line = obj_full_own_lines[line_ind]
            edge_pt = obj_int_pts[edge_pt_ind]
            edge_lines = obj_edge_lines[edge_pt_ind]
            line_edge_point = check_line_edge_point_pepend(line, edge_pt, edge_lines)
            if line_edge_point:
                the_line_edge_points.append([edge_pt, line])
                count_line_edge_points += 0
    if count_parallel_lines + count_line_edge_points > 0:
        return 1
    else:
        return 0

def get_stable_configs_feature(obj_props_input):

    obj_props = compute_obj_props(obj_props_input)

    all_obj_own_lines, all_obj_int_pts, all_obj_edge_lines, all_obj_full_own_lines, all_obj_full_edge_lines = obj_props

    all_stable_configs = []
    for obj_ind in range (num_objs):
        stable_config_params = []
        stable_config_params.append(all_obj_own_lines[obj_ind])
        stable_config_params.append(all_obj_int_pts[obj_ind])
        stable_config_params.append(all_obj_edge_lines[obj_ind])
        stable_config_params.append(all_obj_full_own_lines[obj_ind])
        stable_config_params.append(all_obj_full_edge_lines[obj_ind])

        num_configs = compute_num_stable_configs(stable_config_params)
        all_stable_configs.append(num_configs)

    stable_config_mean = np.mean(all_stable_configs)
    stable_config_std = np.std(all_stable_configs)

    return stable_config_mean, stable_config_std

def check_line_edge_point_pepend(line, edge_pt, edge_lines, debug_flag=False):

    line_edge_point_pepend = False

    # First find distance between line and point
    pepend_dist = LineString(line).distance(Point(edge_pt))

    SMALL_EPS = 1e-3
    if pepend_dist < SMALL_EPS:
        # Point lies on line
        line_edge_point_pepend = False
    else:
        # There is a chance

        # Find normal to line of interest
        line_vec = np.array(line[1]) - np.array(line[0])
        line_unit_vec = line_vec/np.linalg.norm(line_vec)

        pepend_vec_1, pepend_vec_2 = find_2d_pepend_vec(line_unit_vec)

        line_pt_pepend_vec_1 = pepend_dist*pepend_vec_1
        line_pt_pepend_vec_2 = pepend_dist*pepend_vec_2

        # Fine normal pointing towards point of interest
        # Start from point of interest, add normal vec*dist. If we arrive at line,
        # then that is the wrong normal vector.
        pt_on_line_1 = line_pt_pepend_vec_1 + np.array(edge_pt)
        pt_on_line_2 = line_pt_pepend_vec_2 + np.array(edge_pt)

        d_1 = LineString(line).distance(Point(pt_on_line_1))
        d_2 = LineString(line).distance(Point(pt_on_line_2))

        valid_case = False
        if d_1 < SMALL_EPS:
            # This is the wrong normal vector index
            line_pt_pepend_vec = line_pt_pepend_vec_2.copy()
            valid_case = True
        elif d_2 < SMALL_EPS:
            # This is the wrong normal vector index
            line_pt_pepend_vec = line_pt_pepend_vec_1.copy()
            valid_case = True
        else:
            # Closet point on line from edge_pt is not along a pependicular line.
            line_edge_point_pepend = False

        if valid_case:

            edge_line_vec_1, edge_line_vec_2 = find_edge_line_vectors(edge_lines, edge_pt)
            ############################################################
            angle_1 = find_angle_btw_vectors(line_pt_pepend_vec, edge_line_vec_1)
            angle_2 = find_angle_btw_vectors(line_pt_pepend_vec, edge_line_vec_2)

            if check_acute(angle_1, angle_2):
                # Both angles are acute.
                # Point line case holds true.
                line_edge_point_pepend = True

    return line_edge_point_pepend

def evaluate_grasps_exp(cand_grasp_params, gp_params, obj_params, data_path):
    """ Compares Mujoco with custom simulator for MOG """

    # General grasp params
    cand_grasps = cand_grasp_params[0]
    cand_grasp_areas = cand_grasp_params[1]

    # Params to check failure
    cand_grasp_swept_polygon = cand_grasp_params[2]
    cand_grasp_plate_polygons = cand_grasp_params[3]
    cand_grasp_obj_int_polygons = cand_grasp_params[4]

    # Compute centroid of objects in current state
    obj_state = gp_params[0]

    obj_centroids = []
    obj_polygons = []
    all_obj_pts = []
    for obj_num in range(num_objs):
        # Object points
        obj_pts = obj_state[obj_num]
        all_obj_pts.append(obj_pts)

        # Centroids
        obj_centroid = np.mean(obj_pts, axis=0)
        obj_centroids.append(obj_centroid)

        # Polygons
        hull_object = ConvexHull(obj_pts)
        hull_path = Path(hull_object.points[hull_object.vertices])
        obj_polygon = Polygon(hull_path.to_polygons()[0])
        obj_polygons.append(obj_polygon)

    mj_success = []
    gp_success = []
    gp_conditions = []

    init_grasp_states = []
    count_c1_examples = 0
    count_c2_examples = 0
    MAX_COND_EXAMPLES = 3
    for grasp_index in range(len(cand_grasps)):

        cand_grasp = cand_grasps[grasp_index]
        swept_polygon = cand_grasp_swept_polygon[grasp_index]
        plate_polygons = cand_grasp_plate_polygons[grasp_index]
        obj_int_polygons = cand_grasp_obj_int_polygons[grasp_index]

        sim_grasp_input = []
        sim_grasp_input.append(swept_polygon)
        sim_grasp_input.append(all_obj_pts)
        sim_grasp_input.append(plate_polygons)
        sim_grasp_input.append(obj_int_polygons)
        sim_grasp_input.append(obj_polygons)

        grasp_success_gp, failure_conds = simulate_grasp_gp(sim_grasp_input)

        gp_success.append(grasp_success_gp)
        gp_conditions.append(failure_conds)

        init_grasp_state, final_grasp_state, grasp_success_mj, grasp_X = simulate_full_grasp(obj_params, cand_grasp)

        init_grasp_states.append(init_grasp_state)

        # False Negative
        if grasp_success_mj == 'True' and grasp_success_gp == 'False':
            fig_path = data_path+'false_negatives/sample_{}/'.format(grasp_index)
            subprocess.call(['rm', '-rf', fig_path])
            subprocess.call(['mkdir', '-p', fig_path])
            capture_state_sequence_frames(grasp_X, fig_path)

        if grasp_success_mj == 'True':
            fig_path = data_path+'positives/sample_{}/'.format(grasp_index)
            subprocess.call(['rm', '-rf', fig_path])
            subprocess.call(['mkdir', '-p', fig_path])
            capture_state_sequence_frames(grasp_X, fig_path)

        if count_c1_examples < MAX_COND_EXAMPLES and failure_conds[0] == False:
            fig_path = data_path+'c1_negatives/sample_{}/'.format(grasp_index)
            subprocess.call(['rm', '-rf', fig_path])
            subprocess.call(['mkdir', '-p', fig_path])
            capture_state_sequence_frames(grasp_X, fig_path)
            count_c1_examples += 1

        if count_c2_examples < MAX_COND_EXAMPLES and failure_conds[1] == False:
            fig_path = data_path+'c2_negatives/sample_{}/'.format(grasp_index)
            subprocess.call(['rm', '-rf', fig_path])
            subprocess.call(['mkdir', '-p', fig_path])
            capture_state_sequence_frames(grasp_X, fig_path)
            count_c2_examples += 1

        mj_success.append(grasp_success_mj)

    return mj_success, gp_success, gp_conditions, init_grasp_states

def capture_state_sequence_frames(grasp_X, fig_path):

    # 25 fps (We need 25*3s evenly spaced frames)
    total_num_frames = grasp_X.shape[0]

    total_desired_frames = 25*3.
    frame_spacing = int(total_num_frames/total_desired_frames)
    frame_indices = np.linspace(start=0, stop=int(total_num_frames), num=int(total_desired_frames), endpoint=False)

    frame_count = 0
    for frame_ind in frame_indices:
        x_frame = grasp_X[int(frame_ind)].copy()
        frame_path = fig_path+'frame-{}.png'.format(frame_count)
        capture_grasp_state(x_frame, frame_path)
        frame_count += 1

    return None

def sim_analysis(mj_success, gp_success):
    # Quick analysis
    correct_false_pred = 0
    correct_true_pred = 0
    maybe_pred = 0
    maybe_then_true_pred = 0
    maybe_then_false_pred = 0
    total_false_pred = 0
    total_true_pred = 0
    total_maybe_pred = 0

    false_negative = 0
    false_positive = 0

    num_cand_grasps = len(mj_success)

    for g_ind in range(num_cand_grasps):

        if mj_success[g_ind] == 'False':
            total_false_pred += 1

        if mj_success[g_ind] == 'True':
            total_true_pred +=  1

        if gp_success[g_ind] == 'Maybe':
            total_maybe_pred += 1

        if mj_success[g_ind] == 'False' and gp_success[g_ind] == 'False':
            correct_false_pred += 1
        elif mj_success[g_ind] == 'True' and gp_success[g_ind] == 'True':
            correct_true_pred += 1
        elif mj_success[g_ind] == 'True' and gp_success[g_ind] == 'Maybe':
            maybe_then_true_pred += 1
        elif mj_success[g_ind] == 'False' and gp_success[g_ind] == 'Maybe':
            maybe_then_false_pred += 1
        elif mj_success[g_ind] == 'False' and gp_success[g_ind] == 'True':
            false_positive += 1
            #print ('False positive at:', g_ind)
        elif mj_success[g_ind] == 'True' and gp_success[g_ind] == 'False':
            false_negative += 1
            #print ('False negative at:', g_ind)
        else:
            print ('Case not found, debug!')

    if total_false_pred != 0:
        cfp_percent = (correct_false_pred/total_false_pred)*100
    else:
        cfp_percent = None

    if total_true_pred != 0:
        ctp_percent = (correct_true_pred/total_true_pred)*100
    else:
        ctp_percent = None

    if total_maybe_pred != 0:
        maybe_false_pred_percent = (maybe_then_false_pred/total_maybe_pred)*100
    else:
        maybe_false_pred_percent = None

    if total_maybe_pred != 0:
        maybe_true_pred_percent = (maybe_then_true_pred/total_maybe_pred)*100
    else:
        maybe_true_pred_percent = None

    print ('\n')
    print ('Correct false predictions: {} and {} percent'.format(correct_false_pred, cfp_percent) )
    print ('Correct true predictions: {} and {} percent'.format(correct_true_pred, ctp_percent) )
    print ('Maybe predictions that are false {} and {} percent of maybe preds'.format(maybe_then_false_pred, maybe_false_pred_percent))
    print ('Maybe predictions that are true {} and {} percent of maybe preds'.format(maybe_then_true_pred, maybe_true_pred_percent))

    fp_percent = (false_positive/num_cand_grasps)*100
    fn_percent = (false_negative/num_cand_grasps)*100

    print ('False positives: {} and {} percent of grasps'.format(false_positive, fp_percent))
    print ('False negatives: {} and {} percent of grasps'.format(false_negative, fn_percent))

    return None

def sort_all_grasps(cand_grasps, cand_grasp_areas, gp_params, sort_method='Area'):

    obj_state = gp_params[0].copy()

    SORT_WEIGHT = 1.0
    # Compute centroid of objects in current state
    obj_1_pts = obj_state[0].copy()
    obj_2_pts = obj_state[1].copy()
    try:
        obj_1_centroid = np.mean(obj_1_pts, axis=0)
    except:
        IPython.embed()
    obj_2_centroid = np.mean(obj_2_pts, axis=0)

    # Rank grasp samples
    grasp_ind = np.linspace(start=0, stop=len(cand_grasps), num=len(cand_grasps), endpoint=False)

    #  Sort criteria is the
    grasp_area_sums = np.sum(cand_grasp_areas, axis=1)
    sort_criteria_area = grasp_area_sums/np.max(grasp_area_sums)

    if sort_method == 'Area':
        # Use Area
        sorted_indices = [x for _, x in sorted(zip(sort_criteria_area, grasp_ind), reverse=True)]
    elif sort_method == 'Random':
        # Randomize grasp indices
        sorted_indices = np.random.permutation(len(cand_grasps))
    else:
        print ('Unknown sort method')

    new_cand_grasps = []
    for g_ind in range(len(cand_grasps)):
        new_cand_grasps.append(cand_grasps[int(sorted_indices[g_ind])])

    return new_cand_grasps, sorted_indices

def check_edge_pepend_connection(pts, lines, debug_flag=False):

    if debug_flag:
        IPython.embed()

    pepend_connection = False
    # Each line meeting at point of interest must have an acute angle
    # w.r.t the conecting line.
    edge_lines_a = lines[0]
    edge_lines_b = lines[1]
    edge_pt_a = pts[0]
    edge_pt_b = pts[1]
    edge_a_line_vec_1, edge_a_line_vec_2 = find_edge_line_vectors(edge_lines_a, edge_pt_a)
    edge_b_line_vec_1, edge_b_line_vec_2 = find_edge_line_vectors(edge_lines_b, edge_pt_b)

    # Checking edge a
    con_line_vec_a = np.array(edge_pt_a) - np.array(edge_pt_b)
    angle_a_1 = find_angle_btw_vectors(con_line_vec_a, edge_a_line_vec_1)
    angle_a_2 = find_angle_btw_vectors(con_line_vec_a, edge_a_line_vec_2)

    edge_a_check = False
    if check_acute(angle_a_1, angle_a_2):
        # Edge A is fine
        edge_a_check = True

    # Checking edge b
    con_line_vec_b = np.array(edge_pt_b) - np.array(edge_pt_a)
    angle_b_1 = find_angle_btw_vectors(con_line_vec_b, edge_b_line_vec_1)
    angle_b_2 = find_angle_btw_vectors(con_line_vec_b, edge_b_line_vec_2)

    edge_b_check = False
    if check_acute(angle_b_1, angle_b_2):
        # Edge A is fine
        edge_b_check = True

    if edge_a_check and edge_b_check:
        pepend_connection = True

    return pepend_connection

def check_line_edge_point_pepend(line, edge_pt, edge_lines, debug_flag=False):

    if debug_flag:
        IPython.embed()

    line_edge_point_pepend = False

    # First find distance between line and point
    pepend_dist = LineString(line).distance(Point(edge_pt))

    SMALL_EPS = 1e-3
    if pepend_dist < SMALL_EPS:
        # Point lies on line
        line_edge_point_pepend = False
    else:
        # There is a chance
        # Find normal to line of interest
        line_vec = np.array(line[1]) - np.array(line[0])
        line_unit_vec = line_vec/np.linalg.norm(line_vec)

        pepend_vec_1, pepend_vec_2 = find_2d_pepend_vec(line_unit_vec)

        line_pt_pepend_vec_1 = pepend_dist*pepend_vec_1
        line_pt_pepend_vec_2 = pepend_dist*pepend_vec_2

        # Fine normal pointing towards point of interest
        # Start from point of interest, add normal vec*dist. If we arrive at line,
        # then that is the wrong normal vector.
        pt_on_line_1 = line_pt_pepend_vec_1 + np.array(edge_pt)
        pt_on_line_2 = line_pt_pepend_vec_2 + np.array(edge_pt)

        d_1 = LineString(line).distance(Point(pt_on_line_1))
        d_2 = LineString(line).distance(Point(pt_on_line_2))

        valid_case = False
        if d_1 < SMALL_EPS:
            # This is the wrong normal vector index
            line_pt_pepend_vec = line_pt_pepend_vec_2.copy()
            valid_case = True
        elif d_2 < SMALL_EPS:
            # This is the wrong normal vector index
            line_pt_pepend_vec = line_pt_pepend_vec_1.copy()
            valid_case = True
        else:
            # Closet point on line from edge_pt is not along a pependicular line.
            line_edge_point_pepend = False

        if valid_case:
            edge_line_vec_1, edge_line_vec_2 = find_edge_line_vectors(edge_lines, edge_pt)
            angle_1 = find_angle_btw_vectors(line_pt_pepend_vec, edge_line_vec_1)
            angle_2 = find_angle_btw_vectors(line_pt_pepend_vec, edge_line_vec_2)

            if check_acute(angle_1, angle_2):
                # Both angles are acute.
                # Point line case holds true.
                line_edge_point_pepend = True
                #print ('Line edge point pepend case found!')

                # # Debugging
                # line_xs = [line[0][0], line[1][0]]
                # line_ys = [line[0][1], line[1][1]]
                # edge_line_1 = edge_lines[0]
                # edge_line_2 = edge_lines[1]
                # edge_line_1_xs = [edge_line_1[0][0], edge_line_1[1][0]]
                # edge_line_1_ys = [edge_line_1[0][1], edge_line_1[1][1]]
                # edge_line_2_xs = [edge_line_2[0][0], edge_line_2[1][0]]
                # edge_line_2_ys = [edge_line_2[0][1], edge_line_2[1][1]]
                # plt.cla()
                # plt.plot(line_xs, line_ys, 'k--')
                # plt.plot(edge_line_1_xs, edge_line_1_ys, 'b--')
                # plt.plot(edge_line_2_xs, edge_line_2_ys, 'g--')
                # plt.scatter(edge_pt[0], edge_pt[1], color='b')
                # plt.scatter(pt_on_line_1[0], pt_on_line_1[1], color='r')
                # plt.scatter(pt_on_line_2[0], pt_on_line_2[1], color='g')
                # plt.savefig('test_point_line_1.png')

    return line_edge_point_pepend

def check_acute(angle_1, angle_2):
    SMALL_EPS =  1e-6
    if (angle_1 - m.pi/2.) < SMALL_EPS and (angle_2 - m.pi/2.) < SMALL_EPS:
        # Both angles are acute.
        # Point line case holds true.
        return True
    else:
        return False

def find_edge_line_vectors(edge_lines, edge_pt):

    # There are two edge lines corresponding to one edge point
    # Edge line vector should point to the edge point
    SMALL_EPS = 1e-6

    h_0 = np.linalg.norm(np.array(edge_pt) - np.array(edge_lines[0][0]))
    h_1 = np.linalg.norm(np.array(edge_pt) - np.array(edge_lines[0][1]))

    if h_0 < SMALL_EPS:
        edge_line_vec_1 = np.array(edge_lines[0][0]) - np.array(edge_lines[0][1])
    elif h_1 < SMALL_EPS:
        edge_line_vec_1 = np.array(edge_lines[0][1]) - np.array(edge_lines[0][0])
    else:
        # There are only two possibilities
        #print ('Case not found - debug')
        edge_line_vec_1 = None

    z_0 = np.linalg.norm(np.array(edge_pt) - np.array(edge_lines[1][0]))
    z_1 = np.linalg.norm(np.array(edge_pt) - np.array(edge_lines[1][1]))

    if z_0 < SMALL_EPS:
        edge_line_vec_2 = np.array(edge_lines[1][0]) - np.array(edge_lines[1][1])
    elif z_1 < SMALL_EPS:
        edge_line_vec_2 = np.array(edge_lines[1][1]) - np.array(edge_lines[1][0])
    else:
        # There are only two possibilities
        #print ('Case not found - debug')
        edge_line_vec_2 = None

    return edge_line_vec_1, edge_line_vec_2

def find_angle_btw_vectors(vec_1, vec_2):
    # Return only positive numbers
    unit_vec_1  = vec_1/np.linalg.norm(vec_1)
    unit_vec_2 = vec_2/np.linalg.norm(vec_2)

    # Dot product of vectors
    dot_prod = np.dot(unit_vec_1, unit_vec_2)

    SMALL_EPS = 1e-6
    # Avoid math error at 1 and -1
    if abs(dot_prod - 1.0) < SMALL_EPS:
        angle_btw = 0.
    elif abs (dot_prod + 1.0) < SMALL_EPS:
        angle_btw = m.pi
    else:
        #angle_btw = m.acos(dot_prod)
        angle_btw = np.arccos(dot_prod)

    return angle_btw

def find_2d_pepend_vec(unit_vec) :
    pepend_vec = np.empty_like(unit_vec)
    pepend_vec[0] = -unit_vec[1]
    pepend_vec[1] = unit_vec[0]

    return pepend_vec, -pepend_vec

def check_parallel_lines(line_a, line_b):

    line_a_vec = np.array(np.array(line_a[1]) - np.array(line_a[0]))
    line_b_vec = np.array(np.array(line_b[1]) - np.array(line_b[0]))

    line_a_unit_vec = line_a_vec/np.linalg.norm(line_a_vec)
    line_b_unit_vec = line_b_vec/np.linalg.norm(line_b_vec)

    dot_val = np.dot(line_a_unit_vec, line_b_unit_vec)

    SMALL_EPS = 1e-4
    parallel = False
    if abs(abs(dot_val) - 1) < SMALL_EPS:
        parallel = True

    return parallel

def simulate_full_grasp(obj_params, cand_grasp):

    g1_pos, g2_pos, g1_quat, g2_quat = get_gripper_params(cand_grasp)

    with env.physics.reset_context():
        for obj_ind in range(num_objs):
            obj_state = obj_params[obj_ind].copy()
            obj_name = obj_names[obj_ind]+"_joint"
            env.physics.named.data.qpos[obj_name][0:2] = obj_state[0].copy()
            env.physics.named.data.qpos[obj_name][3:7] = obj_state[1].copy()

        env.physics.named.model.body_quat['left_plate'] = g1_quat.copy()
        env.physics.named.model.body_quat['right_plate'] = g2_quat.copy()
        env.physics.named.model.body_pos['left_plate'][0:2] = g1_pos.copy()
        env.physics.named.model.body_pos['right_plate'][0:2] = g2_pos.copy()
        env.physics.step()

    init_grasp_state = env.physics.get_state()
    final_grasp_state, grasp_success, grasp_X = simulate_grasp(init_grasp_state, cand_grasp)

    return init_grasp_state, final_grasp_state, grasp_success, grasp_X

def baseline_algo(cand_grasps, obj_params, grasp_indices):

    """The baseline algorithm samples grasp candidates
        and tests them in Mujoco until a stable grasp is found.
    """

    count_used_samples = 0
    for grasp_index in range(len(cand_grasps)):
        count_used_samples += 1
        cand_grasp = cand_grasps[int(grasp_indices[grasp_index])]

        init_grasp_state, final_grasp_state, grasp_success, grasp_X = simulate_full_grasp(obj_params, cand_grasp)

        if grasp_success == 'True':
            break

    results = [grasp_success, count_used_samples, int(grasp_indices[grasp_index])]

    return cand_grasp, results

def save_state_fig(dum_state, fig_name='fig.png'):
    set_state(dum_state)
    image_data = env.physics.render(height=480, width=640, camera_id=0)
    img = Image.fromarray(image_data, 'RGB')
    img.save(fig_name)
    return None

def capture_grasp_state(grasp_state, fig_path='./'):
    set_state(grasp_state)
    image_data = env.physics.render(height=480, width=640, camera_id=0)
    img = Image.fromarray(image_data, 'RGB')
    img.save(fig_path)
    return None

def check_grasp_success(final_state, min_stable_dist, indiv_min, grasp_cand):

    # Uses the diameter function to check grasp success
    # (faster than robot lifting)
    lpy = final_state[0]
    rpy = final_state[1]

    curr_grip_dist = G_STROKE - (abs(lpy) + abs(rpy))
    dist_btw_plates = curr_grip_dist - 2*G_WIDTH

    eps_val = np.min(indiv_min)/2.

    stable_low = min_stable_dist - eps_val
    stable_high = min_stable_dist + eps_val

    # # And via intersection area (meaning object not in gripper)
    robot_final_coords = get_final_robot_coords(grasp_cand, dist_btw_plates)
    grippers_hull = ConvexHull(robot_final_coords[-1])
    grippers_hull_path = Path(grippers_hull.points[grippers_hull.vertices])
    sp_final_polygon = Polygon(grippers_hull_path.to_polygons()[0])

    #if final_state != []:
    set_state(final_state)

    final_obj_state, pt_list = get_object_state()

    obj_final_polygons = []
    for obj_ind in range(num_objs):
        try:
            hull_object = ConvexHull(final_obj_state[obj_ind])
        except:
            print ('Object on its side - fell off!')
        hull_path = Path(hull_object.points[hull_object.vertices])
        obj_polygon = Polygon(hull_path.to_polygons()[0])
        obj_final_polygons.append(obj_polygon)

    obj_int_polygons = []
    for obj_ind in range(num_objs):
        obj_final = obj_final_polygons[obj_ind]
        obj_fin_int = sp_final_polygon.intersection(obj_final)
        obj_int_polygons.append(obj_fin_int)

    int_area_cond = int_area_condition_check(obj_int_polygons)

    if dist_btw_plates > stable_low and int_area_cond:
        grasp_success = 'True'
    else:
        grasp_success = 'False'

    return grasp_success

def simulate_grasp(x_0, grasp_cand):
    global env

    set_state(x_0)

    min_stable_dist, indiv_min = compute_min_stable_distance()

    grasp_time = 3 # seconds
    num_grasp_steps = int(grasp_time/SIM_TIMESTEP)
    X = np.zeros((num_grasp_steps, x_0.shape[0]))
    for step in range(num_grasp_steps):
        env.physics.data.ctrl[:] = np.array([-1., 1.])
        env.physics.step()
        X[step] = env.physics.get_state()

    final_state = env.physics.get_state()
    grasp_success = check_grasp_success(final_state, min_stable_dist, indiv_min, grasp_cand)

    return final_state, grasp_success, X

def gen_cand_grasps(init_state=[], plot_all_grasps=False, plot_path='./', use_bbox='ch'):

    # State is the position of all points that make up an object.
    #if init_state != []:
    set_state(init_state)

    obj_state, pt_list = get_object_state()

    hull = ConvexHull(np.array(pt_list))
    hulls = []
    obj_polygons = []
    hulls.append(hull)
    for obj_ind in range(num_objs):
        hull_object = ConvexHull(obj_state[obj_ind])
        hull_path = Path(hull_object.points[hull_object.vertices])
        hulls.append(hull_object)
        obj_polygon = Polygon(hull_path.to_polygons()[0])
        obj_polygons.append(obj_polygon)

    uniform_position_samples, short_orn, long_orn, short_vec, long_vec, ct_pt, bbox_pts = generate_uniform_samples(np.array(pt_list), hulls, plot_path=plot_path)

    bbox_hull = ConvexHull(bbox_pts)
    if use_bbox == 'ch':
        orientations_per_point = np.linspace(start=0, stop=m.pi, num=N_orns, endpoint=False)
        # Generate grasp candidates
        grasp_candidates = []
        for pt in uniform_position_samples:
            for orn in orientations_per_point:
                grasp_cand = [pt, orn]
                grasp_candidates.append(grasp_cand)

    elif use_bbox == 'short':
        orn = short_orn
        grasp_candidates = []
        for pt in uniform_position_samples:
            grasp_cand = [pt, orn]
            grasp_candidates.append(grasp_cand)

    else:
        # print ('We got here')
        orn = long_orn
        grasp_candidates = []
        num_midline_pts = 1
        mideline_vec = short_vec/np.linalg.norm(short_vec)
        midline_pts = []
        pt_steps = np.linspace(start=-G_WIDTH, stop=G_WIDTH, num=num_midline_pts)
        for pt_ind in range(num_midline_pts):
            pt_step = pt_steps[pt_ind]
            # pt = ct_pt - pt_step*mideline_vec
            pt = ct_pt
            midline_pts.append(pt)
            # print (pt)

        for pt in midline_pts:
            grasp_cand = [pt, orn]
            grasp_candidates.append(grasp_cand)

    valid_grasps = []
    valid_grasp_hulls = []
    valid_grasp_areas = []
    valid_grasp_swept_polygons = []
    valid_grasp_plate_polygons = []
    valid_grasp_obj_int_polygons = []
    valid_grasp_grippers_hull_paths = []
    plot_grasp_hulls = []
    for grasp_cand in grasp_candidates:
        # Compute intersecting area between gripper swept area and objects.
        A_objs, grasp_hulls, collision, obj_int_polygons, s_poly, plate_polygons, grippers_hull_path = compute_grasp_params(grasp_cand, obj_polygons)

        plot_grasp_hulls.append(grasp_hulls)
        # Eliminate grasps in collision
        if collision == False:
            valid_grasps.append(grasp_cand)
            valid_grasp_hulls.append(grasp_hulls)
            valid_grasp_areas.append(A_objs)
            valid_grasp_swept_polygons.append(s_poly)
            valid_grasp_plate_polygons.append(plate_polygons)
            valid_grasp_obj_int_polygons.append(obj_int_polygons)
            valid_grasp_grippers_hull_paths.append(grippers_hull_path)

    # Plot the grasps
    plot_path = './grasping/'
    if plot_all_grasps and use_bbox == 'long':
        fig_path = plot_path+'grasp_figs/bbox_long/'
        subprocess.call(['mkdir', '-p', fig_path])
        # for grasp in range(len(valid_grasps)):
        for grasp in range(len(grasp_candidates)):

            # grasp_hull = valid_grasp_hulls[grasp]
            grasp_hull = plot_grasp_hulls[grasp]
            hull_left_plate = grasp_hull[0]
            hull_right_plate = grasp_hull[1]
            hull_swept_area = grasp_hull[2]

            all_hulls = hulls[1:].copy()
            all_hulls.append(hull_left_plate)
            all_hulls.append(hull_right_plate)
            all_hulls.append(hull_swept_area)

            plot_grasp(grasp_candidates[grasp], all_hulls, grasp_index=grasp, p_path=fig_path, bbox_hull=bbox_hull)
    # IPython.embed()
    valid_grasp_params = [valid_grasps, valid_grasp_areas, valid_grasp_swept_polygons,
                        valid_grasp_plate_polygons, valid_grasp_obj_int_polygons,
                        valid_grasp_grippers_hull_paths]

    return valid_grasp_params, obj_state

def int_area_condition_check(obj_int_polygons):

    int_area_cond = True
    for obj_ind in range(num_objs):
        if obj_int_polygons[obj_ind].area <= 0:
            int_area_cond = False
            break

    return int_area_cond

def get_gripper_params(grasp_cand):

    grasp_center = grasp_cand[0]
    grasp_orn = grasp_cand[1]
    grip_stroke = G_STROKE

    gripper_l = grip_stroke/2.
    lp_center_pt = grasp_center + gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])
    rp_center_pt = grasp_center - gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])
    lp_quat = Quaternion(axis=[0, 0, 1], angle=-(m.pi/2.-grasp_orn)).elements
    rp_quat = lp_quat.copy()

    return lp_center_pt, rp_center_pt, lp_quat, rp_quat

def compute_grasp_params(grasp_cand, obj_polygons):

    robot_coords = get_robot_coords(grasp_cand)

    left_plate_hull = ConvexHull(robot_coords[0])
    right_plate_hull = ConvexHull(robot_coords[1])
    grippers_hull = ConvexHull(robot_coords[-1])

    # Find grasp midline
    left_plate_coords = robot_coords[0]
    right_plate_coords = robot_coords[1]
    swept_area_coords = robot_coords[2]

    left_plate_pts = []
    right_plate_pts = []
    small_eps = 1e-6
    for coord in swept_area_coords:
        for coord_l in left_plate_coords:
            if np.linalg.norm(coord-coord_l) <= small_eps:
                left_plate_pts.append(coord)
        for coord_r in right_plate_coords:
            if np.linalg.norm(coord-coord_r) <= small_eps:
                right_plate_pts.append(coord)

    # Find intersecting areas
    grippers_hull_path = Path(grippers_hull.points[grippers_hull.vertices])
    swept_gripper_polygon = Polygon(grippers_hull_path.to_polygons()[0])

    # Get intersection areas
    A_objs = []
    obj_int_polygons = []
    for obj_ind in range(num_objs):
        obj_intersect = swept_gripper_polygon.intersection(obj_polygons[obj_ind])
        A_objs.append(obj_intersect.area)
        obj_int_polygons.append(obj_intersect)

    # Check plate collisions
    left_plate_path = Path(left_plate_hull.points[left_plate_hull.vertices])
    right_plate_path = Path(right_plate_hull.points[right_plate_hull.vertices])

    left_plate_polygon = Polygon(left_plate_path.to_polygons()[0])
    right_plate_polygon = Polygon(right_plate_path.to_polygons()[0])

    lp_intersects = []
    rp_intersects = []
    for obj_ind in range(num_objs):
        lp_intersect = left_plate_polygon.intersects(obj_polygons[obj_ind])
        rp_intersect = right_plate_polygon.intersects(obj_polygons[obj_ind])
        lp_intersects.append(lp_intersect)
        rp_intersects.append(rp_intersect)

    all_intersects = []
    all_intersects.extend(lp_intersects)
    all_intersects.extend(rp_intersects)
    collides = False
    for intersect in all_intersects:
        if intersect:
            collides = True
            break

    grasp_specific_hulls = [left_plate_hull, right_plate_hull, grippers_hull]
    plate_polygons = [left_plate_polygon, right_plate_polygon]

    return A_objs, grasp_specific_hulls, collides, obj_int_polygons, swept_gripper_polygon, plate_polygons, grippers_hull_path

def generate_lines_from_pts(pts):

    lines = []
    for pt in range(len(pts)-1):
        line = [pts[pt], pts[pt+1]]
        lines.append(line)

    # Add the last line
    line = [pts[-1], pts[0]]

    lines.append(line)

    return lines

def wrap_points(pts):
    pts_arr = np.array(pts)
    num_coords = pts_arr.shape[1]

    wrapped_points = []
    for coord in range(num_coords):
        wrapped_points.append([pts_arr[0][coord], pts_arr[1][coord]])

    return wrapped_points

def get_robot_state():

    num_plate_pts = 4
    left_plate_pts = []
    right_plate_pts = []
    for pt in range(num_plate_pts):
        # Left plate
        left_pt_name = 'left_plate_point_{}'.format(pt+1)
        left_pt_pos = env.physics.named.data.site_xpos[left_pt_name][0:2]
        left_plate_pts.append(list(left_pt_pos))

        # Right plate
        right_pt_name = 'right_plate_point_{}'.format(pt+1)
        right_pt_pos = env.physics.named.data.site_xpos[right_pt_name][0:2]
        right_plate_pts.append(list(right_pt_pos))

    return [left_plate_pts, right_plate_pts]

def generate_uniform_samples(pos, hulls, create_plots=False, plot_path='./', use_bbox=False):

    hull = hulls[0]
    all_hull_objects = hulls[1:]

    # Generate an outer bounding box that encloses the convex hull
    outer_bbox = [hull.min_bound, hull.max_bound]

    hull_pts = hull.points
    bbox_corners, ct_pt = get_oriented_bbox(hull_pts)
    xvec = bbox_corners[1] - bbox_corners[0]
    orn_1 = m.atan2(xvec[1], xvec[0])

    yvec = bbox_corners[2] - bbox_corners[1]
    orn_2 = m.atan2(yvec[1], yvec[0])

    # yyvec = bbox_corners[3] - bbox_corners[2]
    # orn_3 = m.atan2(yyvec[1], yyvec[0])

    # print ('Orientation 1', orn_1)
    # print ('Orientation 2', orn_2)
    # print ('Orientation 3', orn_3)
    short_orn = orn_1
    long_orn = orn_2

    # Divide up the outer bounding box into evenly-spaced grids.
    # Number of grids is N_grids
    N_grids = N_cols*N_rows

    # Generate all the sub bounding boxes
    x_axes_vals = np.linspace(start=outer_bbox[0][0], stop=outer_bbox[1][0], num=N_cols+1)
    y_axes_vals = np.linspace(start=outer_bbox[0][1], stop=outer_bbox[1][1], num=N_rows+1)

    bboxes = []
    bbox_starts = []
    for x_val in range(N_cols):
        for y_val in range(N_rows) :
            coord_1 = [x_axes_vals[x_val], y_axes_vals[y_val]]
            coord_2 = [x_axes_vals[x_val+1], y_axes_vals[y_val+1]]
            bbox_extents = np.array([coord_1, coord_2])
            bboxes.append(bbox_extents)
            bbox_starts.append(coord_1)

    rand_points = np.empty((N_grids, 2))
    # Generate one sample per grid (grid center?)
    for grid in range(N_grids):
        grid_start_point = np.array(bbox_starts[grid])
        rand_points[grid] = grid_start_point + np.array([bboxes[grid][1] - bboxes[grid][0]])/2.

    hull_path = Path(hull.points[hull.vertices])

    ch_points = []
    ch_polygon = Polygon(hull_path.to_polygons()[0])
    for pt_index in range(N_grids):
        # Discard random point if it is not inside the convex hull
        pt = rand_points[pt_index].copy()
        # if use_bbox:
        #     ch_points.append(pt)
        # else:
        if ch_polygon.contains(Point(pt)):
            ch_points.append(pt)

    ch_points_array = np.array(ch_points)

    mj_coords = True

    # plt.scatter(-pos[:, 1], pos[:, 0], marker='o',  c='blue', alpha = 1)

    # cls = ['r', 'g', 'y']
    #
    # count = 0
    # plot_path = "./"
    #
    # for obj_ind in range(num_objs):
    #     hull_object = all_hull_objects[obj_ind]
    #     for simplex in hull_object.simplices:
    #             plt.plot(-hull_object.points[simplex, 1], hull_object.points[simplex, 0], cls[count])
    #     count += 1
    # plt.ylim(1.85-plot_width, 1.85+plot_width)
    # plt.xlim(0.-plot_width, 0.+plot_width)
    # plt.savefig(plot_path+"uniform_samples_init.png", dpi = 300)
    #
    # for simplex in hull.simplices:
    #         plt.plot(-hull.points[simplex, 1], hull.points[simplex, 0], 'k--')
    #
    # plt.savefig(plot_path+"uniform_samples_ch.png", dpi = 300)
    #
    # plt.scatter(-ch_points_array[:, 1],ch_points_array[:, 0], marker='.',  c='blue', alpha = 0.5)
    #
    # plt.savefig(plot_path+"uniform_samples_scatter.png", dpi = 300)
    # plt.cla()

    # IPython.embed()

    return ch_points_array, short_orn, long_orn, xvec, yvec, ct_pt, bbox_corners

def get_oriented_bbox(a):
    # a  = np.array([(3.7, 1.7), (4.1, 3.8), (4.7, 2.9), (5.2, 2.8), (6.0,4.0), (6.3, 3.6), (9.7, 6.3), (10.0, 4.9), (11.0, 3.6), (12.5, 6.4)])

    ca = np.cov(a,y = None,rowvar = 0,bias = 1)

    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)

    # fig = plt.figure(figsize=(12,12))
    # ax = fig.add_subplot(111)
    # ax.scatter(a[:,0],a[:,1])

    #use the inverse of the eigenvectors as a rotation matrix and
    #rotate the points so they align with the x and y axes
    ar = np.dot(a,np.linalg.inv(tvect))

    # get the minimum and maximum x and y
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff = (maxa - mina)*0.5

    # the center is just half way between the min and max xy
    center = mina + diff

    #get the 4 corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]],center+[-diff[0],-diff[1]]])

    #use the the eigenvectors as a rotation matrix and
    #rotate the corners and the centerback
    corners = np.dot(corners,tvect)
    center = np.dot(center,tvect)

    # ax.scatter([center[0]],[center[1]])
    # ax.plot(corners[:,0],corners[:,1],'-')

    # plt.axis('equal')
    # plt.show()

    return corners, center

def get_object_state():

    all_obj_pts = []
    pt_list = []
    for obj_ind in range(num_objs):
        obj_pts  = []
        obj_name = obj_names[obj_ind]
        obj_num_pts = all_obj_num_pts[obj_ind]
        for pt in range(obj_num_pts):
            pt_name = obj_name+'_point_{}'.format(pt+1)
            pt_pos = env.physics.named.data.site_xpos[pt_name][0:2].copy()
            obj_pts.append(list(pt_pos))
            pt_list.append(list(pt_pos))

        all_obj_pts.append(obj_pts)

    return all_obj_pts, pt_list

def get_robot_coords(grasp_cand):

    grasp_center = grasp_cand[0]
    grasp_orn = grasp_cand[1]
    grip_stroke = G_STROKE

    gripper_l = grip_stroke/2.
    lp_center_pt = grasp_center + gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])
    rp_center_pt = grasp_center - gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])

    lp_vec = lp_center_pt - grasp_center
    rp_vec = rp_center_pt - grasp_center

    lp_unit_vec = lp_vec/np.linalg.norm(lp_vec)
    rp_unit_vec = rp_vec/np.linalg.norm(rp_vec)

    lp_pepend_vec = get_pepend_vectors(lp_unit_vec)[0]
    rp_pepend_vec = get_pepend_vectors(rp_unit_vec)[0]

    left_plate_coords = plate_coords(lp_center_pt, lp_unit_vec, lp_pepend_vec)
    right_plate_coords = plate_coords(rp_center_pt, rp_unit_vec, rp_pepend_vec)

    swept_area_coords = sp_coords(grasp_center, lp_unit_vec, lp_pepend_vec, grip_stroke)

    return [left_plate_coords, right_plate_coords, swept_area_coords]

def get_final_robot_coords(grasp_cand, grip_stroke):

    grasp_center = grasp_cand[0]
    grasp_orn = grasp_cand[1]

    gripper_l = grip_stroke/2.
    lp_center_pt = grasp_center + gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])
    rp_center_pt = grasp_center - gripper_l*np.array([m.cos(grasp_orn), m.sin(grasp_orn)])

    lp_vec = lp_center_pt - grasp_center
    rp_vec = rp_center_pt - grasp_center

    lp_unit_vec = lp_vec/np.linalg.norm(lp_vec)
    rp_unit_vec = rp_vec/np.linalg.norm(rp_vec)

    lp_pepend_vec = get_pepend_vectors(lp_unit_vec)[0]
    rp_pepend_vec = get_pepend_vectors(rp_unit_vec)[0]

    left_plate_coords = plate_coords(lp_center_pt, lp_unit_vec, lp_pepend_vec)
    right_plate_coords = plate_coords(rp_center_pt, rp_unit_vec, rp_pepend_vec)

    swept_area_coords = sp_coords(grasp_center, lp_unit_vec, lp_pepend_vec, grip_stroke)

    return [left_plate_coords, right_plate_coords, swept_area_coords]

def get_pepend_vectors(unit_vec):

    # Find pependicular vectors
    v_x = unit_vec[0]
    v_y = unit_vec[1]
    v_x_pepend = np.sqrt(1 - ((v_x**2)/(v_x**2 + v_y**2)))
    v_y_pepend = np.sqrt((v_x**2)/(v_x**2 + v_y**2))

    # There are two possibilities either 0 or 1
    if abs(np.dot(unit_vec[0:2],  np.array([v_x_pepend, v_y_pepend]))) <= 0.5:
        pepend_vector_1 = np.array([v_x_pepend, v_y_pepend])

    elif abs(np.dot(unit_vec[0:2],  np.array([-v_x_pepend, v_y_pepend]))) <= 0.5:
        pepend_vector_1 = np.array([-v_x_pepend, v_y_pepend])
    else:
        pepend_vector_1 = np.array([v_x_pepend, v_y_pepend])

    pepend_vector_2 = -pepend_vector_1

    return pepend_vector_1, pepend_vector_2

def plate_coords(p_center_pt, p_unit_vec, p_pepend_vec):

    """ Convert to plate coordinates """

    #print (np.linalg.norm(p_pepend_vec))

    p_coord_1 = p_center_pt + (G_WIDTH)*p_unit_vec + (G_LEN)*p_pepend_vec
    p_coord_2 = p_center_pt + (G_WIDTH)*p_unit_vec - (G_LEN)*p_pepend_vec
    p_coord_3 = p_center_pt - (G_WIDTH)*p_unit_vec + (G_LEN)*p_pepend_vec
    p_coord_4 = p_center_pt - (G_WIDTH)*p_unit_vec - (G_LEN)*p_pepend_vec

    return [p_coord_1, p_coord_2, p_coord_3, p_coord_4]

def sp_coords(p_center_pt, p_unit_vec, p_pepend_vec, grip_stroke):

    p_coord_1 = p_center_pt + (grip_stroke/2.-G_WIDTH)*p_unit_vec + (G_LEN)*p_pepend_vec
    p_coord_2 = p_center_pt + (grip_stroke/2.-G_WIDTH)*p_unit_vec - (G_LEN)*p_pepend_vec
    p_coord_3 = p_center_pt - (grip_stroke/2.-G_WIDTH)*p_unit_vec + (G_LEN)*p_pepend_vec
    p_coord_4 = p_center_pt - (grip_stroke/2.-G_WIDTH)*p_unit_vec - (G_LEN)*p_pepend_vec

    return [p_coord_1, p_coord_2, p_coord_3, p_coord_4]

def plot_grasp(grasp_cand, hulls, grasp_index=0, p_path='./', bbox_hull=[]):

    grasp_center = grasp_cand[0]
    grasp_orn = grasp_cand[1]

    obj_hulls = hulls[0:num_objs]

    hull_left_plate = hulls[-3]
    hull_right_plate = hulls[-2]
    hull_swept_area = hulls[-1]

    mj_coords =  True

    if mj_coords:
        for obj_ind in range(num_objs):
            hull_object = obj_hulls[obj_ind]
            for simplex in hull_object.simplices:
                    plt.plot(-hull_object.points[simplex, 1], hull_object.points[simplex, 0], obj_colors[obj_ind])
        plt.ylim(grasp_center[0]-plot_width, grasp_center[0]+plot_width)
        plt.xlim(grasp_center[1]-plot_width, grasp_center[1]+plot_width)
        plt.savefig(p_path+"grasp_{}_init.png".format(grasp_index), dpi=150)

        for simplex in bbox_hull.simplices:
                plt.plot(-bbox_hull.points[simplex, 1], bbox_hull.points[simplex, 0], 'b--')
        plt.savefig(p_path+"grasp_{}_bbox.png".format(grasp_index), dpi=150)

        for simplex in hull_left_plate.simplices:
                plt.plot(-hull_left_plate.points[simplex, 1], hull_left_plate.points[simplex, 0], 'k', linewidth="3")
        for simplex in hull_right_plate.simplices:
                plt.plot(-hull_right_plate.points[simplex, 1], hull_right_plate.points[simplex, 0], 'k', linewidth="3")
        for simplex in hull_swept_area.simplices:
                plt.plot(-hull_swept_area.points[simplex, 1], hull_swept_area.points[simplex, 0], 'k--')

        plt.scatter(-grasp_center[1], grasp_center[0], marker='+',  c='blue', alpha = 0.5)
        plt.savefig(p_path+"grasp_{}.png".format(grasp_index), dpi=150)

        plt.cla()

def exp_1_gen_cands():
    global env
    global min_stable_distance

    print ('+++ Gen Cands Data for objects ++++ : ', obj_names)

    base_data_path = './mog_data_exp_1_{}/objs'.format(num_objs)

    for obj_ind in range(num_objs):
        base_data_path = base_data_path + '_' + obj_names[obj_ind]

    rand_init_num = np.random.randint(0, 6000)
    data_path = base_data_path + '/'
    cand_gen_path = data_path+'cand_gen_{}/'.format(rand_init_num)
    sim_data_path = base_data_path+'/sim_compare/'
    subprocess.call(['rm', '-rf', sim_data_path])
    subprocess.call(['mkdir', '-p', sim_data_path])

    subprocess.call(['rm', '-rf', cand_gen_path])
    subprocess.call(['mkdir', '-p', cand_gen_path])

    all_true_bbox_positive = []
    all_true_bbox_negative = []
    all_grasp_exists = []
    all_grasp_success_bbox = []
    #+++++++++++++++++++++++Sample an initial state++++++++++++++++++++++++++
    for sample_scene in range(NUM_SAMPLE_SCENES):

        # print ('Scene number {}'.format(sample_scene))
        sim_scene_path = sim_data_path+'scene_{}/'.format(sample_scene)
        subprocess.call(['rm', '-rf', sim_scene_path])
        subprocess.call(['mkdir', '-p',sim_scene_path])
        invalid_state = True
        MAX_TRIALS = 1000
        num_trials = 0

        desired_table_center = np.array([1.85, 0.0])

        while invalid_state and num_trials <= MAX_TRIALS:

            obj_quats = []
            obj_positions = []
            full_obj_state = []

            store_pos = 0
            eps_val = 1e-3
            for obj_ind in range(num_objs):
                obj_theta = np.random.uniform(low=-m.pi/10, high=m.pi/10)
                obj_quat = Quaternion(axis=[0,0,1], angle=obj_theta).elements

                if obj_ind == 0:
                    rel_val = eps_val + w_list[obj_ind]
                else:
                    rel_val = eps_val + w_list[obj_ind-1] + w_list[obj_ind]
                store_pos += rel_val

                obj_pos = desired_table_center - np.array([0, 0.04]) + np.array([0., store_pos])

                # Add variation in x
                y_var = np.random.uniform(low=-0.015, high=0.015)
                obj_pos[0] += y_var

                obj_quats.append(obj_quat)
                obj_positions.append(obj_pos)
                full_obj_state.append([obj_pos, obj_quat])

            # Update object states
            with env.physics.reset_context():
                for obj_ind in range(num_objs):
                    obj_name = obj_names[obj_ind]+"_joint"
                    obj_state =  full_obj_state[obj_ind].copy()
                    env.physics.named.data.qpos[obj_name][0:2] = obj_state[0].copy()
                    env.physics.named.data.qpos[obj_name][3:7] = obj_state[1].copy()

                env.physics.step()

            init_state = env.physics.get_state()
            invalid_state = check_valid_init_state()
            num_trials += 1

        if num_trials <= MAX_TRIALS:
            #print ('found sample!')
            pass

        if sample_scene == 0:
            # Compute this only once (only dependent on the objects not their state)
            min_stable_distance, _ = compute_min_stable_distance()

        grasp_exists, grasp_success_bbox =  exp_1(init_state , full_obj_state, plot_path=sim_scene_path)

        all_grasp_exists.append(grasp_exists)
        if grasp_exists == 1.:
            all_grasp_success_bbox.append(grasp_success_bbox)


    np.save(cand_gen_path+'all_grasp_exists', all_grasp_exists)
    np.save(cand_gen_path+'all_grasp_success_bbox', all_grasp_success_bbox)

    print ('Percentage d.n.e:', 1 - np.sum(all_grasp_exists)/len(all_grasp_exists))
    if len(all_grasp_success_bbox) > 0:
        print ('Percentage True if exists:', np.sum(all_grasp_success_bbox)/len(all_grasp_success_bbox))

    return None

def exp_1(init_state, full_obj_state, plot_path='./'):
    """ Generating candidate grasps through three different methods """

    # convex_hull_cands, _ = gen_cand_grasps(init_state, use_bbox='ch',  plot_path=plot_path)
    # bbox_short_cands, _ = gen_cand_grasps(init_state, use_bbox='short', plot_path=plot_path)
    bbox_long_cands, _ = gen_cand_grasps(init_state, use_bbox='long', plot_all_grasps='True', plot_path=plot_path)

    true_bbox_negative = 0
    true_bbox_positive = 0
    true_ch_grasps = 0

    num_bbox_grasp_samples = len(bbox_long_cands[0])
    if num_bbox_grasp_samples == 0:
        grasp_exists = 0
        grasp_success_bbox = None
    else:
        grasp_exists = 1
        bbox_long = bbox_long_cands[0][0]
        init_grasp_state, _, bbox_long_success, _ = simulate_full_grasp(full_obj_state, bbox_long)
        if bbox_long_success == 'True' or bbox_long_success == True:
            grasp_success_bbox = 1
        else:
            grasp_success_bbox = 0

    return grasp_exists, grasp_success_bbox


def main_exp():

    # Generate a candidate grasp with 3 methods

    # Method 1: bbox short or long else random (record)

    # Method 2: Necc conds + ranking

    # Method 3: random

    # Execute candidate grasp and get T/F data

    # Record vertices and features as state representations

    pass

initialize_physics_env()

if __name__ == '__main__':
    #sim_mog_exp()
    gen_dataset()
    # exp_1_gen_cands()
