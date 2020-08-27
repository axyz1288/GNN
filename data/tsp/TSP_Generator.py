import time
import argparse
import numpy as np
from concorde.tsp import TSPSolver

def generate_data(coordinate_file, tour_file, num_samples, num_nodes, node_dim): 
    set_nodes_coord = np.random.random([num_samples, num_nodes, node_dim])
    with open(coordinate_file, "w") as f_coordinate:
        with open(tour_file, "w") as f_tour:
            for nodes_coord in set_nodes_coord:
                solver = TSPSolver.from_data(nodes_coord[:,0], nodes_coord[:,1], norm="GEO")  
                solution = solver.solve()
                np.savetxt(f_coordinate, nodes_coord)
                np.savetxt(f_tour, [solution.tour])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--node_dim", type=int, default=2)
    opts = parser.parse_args()
    
    start_time = time.time()
    
    # training set
    generate_data(f"train_tsp{opts.num_nodes}_coordinate.txt", f"train_tsp{opts.num_nodes}_tour.txt",
                  int(opts.num_samples * 0.8), opts.num_nodes, opts.node_dim)
    
    # validation set
    generate_data(f"valid_tsp{opts.num_nodes}_coordinate.txt", f"valid_tsp{opts.num_nodes}_tour.txt",
                  int(opts.num_samples * 0.2), opts.num_nodes, opts.node_dim)
                
    end_time = time.time() - start_time
    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.num_nodes}.")
    print(f"Total time: {end_time/60:.1f}min")
    print(f"Average time: {(end_time/60)/opts.num_samples:.1f}min")