# Dependencies
The following libraries are required for our implementation:
- mpi.h
- math.h
- stdio.h
- stdlib.h
- string.h
- unistd.h

# Instructions
Compile and execute the program using the following commands:

```bash
mpicc PageRank.c -o mpi_application
```

Run the program with the following command:
	
```bash
mpirun -np {procs} ./mpi_application {n} {filename}”
```

Where:
procs - number of processes
n - number of nodes in the pageRank data.
filename - text file containing all of the directed node edges (outgoing page links)


# Description
Firstly, the program reads through an input file in parallel. This is accomplished by splitting the workload (based on file size) equally among the processors (other than process 0, which serves as a `master` process). The master process will initialise an n x n matrix to all zeros. The `slave` processors will read character by character through their section of input, telling the `master` process where to enter a '1' in the adjacency matrix. This approach allows the process to be parallelized for the file reading, while not having to maintain a shared matrix among all processors.

The algorithm then splits the adjacency matrix into chunks of columns to be sent to each process using `MPI_Scatter`. Like in many other programming languages, C stores 2D arrays in row-major order. This creates an issue when splitting a matrix based on columns. To alleviate this, we create an `MPI_Datatype` to be able to then split based on columns. Two types are created, one used to handle the entire n x n matrix, and one used to handle the submatrices scattered to each process (n x number of procs).

Each process then converts their share of columns to normal form. This is done by firstly looping through a column and counting the number of 1’s in the column. It then loops again, setting each 1 to \[1/# of 1’s\]. Each process performs this process for all their respective columns. If there are no ones in a column, all elements of that column are set to 1 / n (# of nodes). 

These columns, while still being kept in their scattered form for each process, are then converted to a google matrix. The google function performs the google calculation ($`G[i][j] = D*S[i][j] + (1-D)*(1/N)`$) for each element of the normalised matrix. MPI_Gather is then used to gather the chunks of the google array (`sub_goog_mat`) together from all procs into an array named ‘goog_mat’ on proc rank 0 (Master Proc). 

We then move on to the PageRank calculation function. The google matrix is multiplied by the current iteration of the PageRank vector (all elements are initially set to 1/N), with the old PageRank vector still stored. This multiplication is done in parallel within the `dot_prod` function, by using `MPI_Scatter` to scatter the relevant rows of the matrix, and the relevant vector elements. Each process multiplies its respective chunks, and results are gathered using MPI_Gather back to the master process. In the master process, we then check the average difference between the elements of the old vector and the elements of the new vector. When we find that the difference is minimal (below a given number deemed adequate enough), we consider convergence reached, and we have our final PageRank vector. During this time, the other processes wait for a broadcast from the master node. The master node will broadcast to the other nodes if another iteration is needed, or if convergence has been reached. If convergence has been reached, all processes break out of the loop, and the vector is returned. If not, then the process is repeated. This is repeated until convergence, or until a set number of maximum iterations. This is done to account for the case in which the calculation takes too long, or in the case there is an issue with the numbers that might never allow convergence to be reached. 

Once convergence is reached and we have the final PageRank vector, the elapsed time is then printed.
