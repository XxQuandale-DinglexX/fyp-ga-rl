from func.datastruct import Point, Machine
from algo.gen_algo_test_copy import GeneticAlgorithm
from algo.gen_algo_rl import RLGeneticAlgorithm
# from visual.visualize import visualize_layout, visualize_layout_generation, create_layout_animation, plot_optimization_progress

def main():
    """Main function demonstrating the optimization"""
    
    # Define machines with different geometries
    machines = [
        Machine(id=1, shape='rectangle', width=4.0, height=3.0, 
                access_point=Point(1.5, 0)),
        Machine(id=2, shape='l_shape', width=5.0, height=4.0, 
                access_point=Point(-1.0, 1.0),
                l_cutout_width=2.0, l_cutout_height=2.0),
        Machine(id=3, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
        Machine(id=4, shape='l_shape', width=4.5, height=3.5, 
                access_point=Point(1.0, -0.5),
                l_cutout_width=1.5, l_cutout_height=1.5),
        Machine(id=5, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=6, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=7, shape='l_shape', width=3.5, height=5.5, 
                access_point=Point(1.0, -1.0),
                l_cutout_width=1, l_cutout_height=2),
        Machine(id=8, shape='rectangle', width=3.5, height=2.5, 
                access_point=Point(0, -1.0)),
    ]
    
    # Define sequence of operations
    sequence = [1,2,3,4,5,6,7,8]  # Robot visits machines in this order
    
    # Robot starting position (center of workspace)
    robot_position = Point(0, 0)
    
    # Workspace boundaries (min_x, max_x, min_y, max_y)
    workspace_bounds = (-15, 15, -15, 15)
    
    print("=== Robotic Workcell Layout Optimization ===")
    print(f"Machines: {len(machines)}")
    print(f"Sequence: {sequence}")
    print(f"Workspace: {workspace_bounds}")
    print()
    
    # Create and run optimization
    optimizer_list = [GeneticAlgorithm, RLGeneticAlgorithm]
    for OptimizerClass in optimizer_list:
        optimizer = OptimizerClass(
            machines=machines,
            sequence=sequence,
            robot_position=robot_position,
            workspace_bounds=workspace_bounds
        )
        
        # Run optimization
        final_machines, best_fitness, results = optimizer.optimize()
        
        # Display results
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Best Fitness: {best_fitness:.2f}")
        print(f"Total Distance: {results['total_distance']:.2f} units")
        print(f"Execution Time: {results['execution_time']:.2f} seconds")
        print(f"Generations: {results['generations']}")
        print()
        
        print("Final Machine Positions:")
        for machine in final_machines:
            print(f"Machine {machine.id}: "
                f"Position=({machine.position.x:.2f}, {machine.position.y:.2f}), "
                f"Rotation={machine.rotation:.1f}°")
        print()

        # Visualize results
        # print("\nGenerating visualization...")
        
        # create_layout_animation(decoded_machines_list, robot_position, sequence, workspace_bounds)
        
        # visualize_layout(final_machines, robot_position, sequence, workspace_bounds)
        # plot_optimization_progress(results)
        
    return final_machines, results

if __name__ == "__main__":
    # Run main optimization
    final_machines, results = main()