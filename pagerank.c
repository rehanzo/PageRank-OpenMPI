#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


typedef struct{
  float prob;
  int page;
} pageRankPair;


float **floatalloc2d(int n, int m, float *data) {
  float **array = (float **)calloc(n, sizeof(float *));
  for (int i = 0; i < n; i++)
    array[i] = &(data[i * m]);


  return array;
}


float *dot_prod(int rank, int nprocs, int n, float *matrix, float *vector) {
  //Function which multiplies a matrix by a vector (helper for PageRank function)
  int rows, remainder, i, j;
  rows = n / nprocs;      // number of rows for each proc
  remainder = n % nprocs; // leftover rows


  int local_rows =
      rows + (rank < remainder ? 1 : 0); // remainder row distributed
  float *local_matrix = (float *)malloc(local_rows * n * sizeof(float));
  float *local_result = (float *)malloc(local_rows * sizeof(float));


  int *sendcounts = malloc(nprocs * sizeof(int));
  int *displs = malloc(nprocs * sizeof(int));


  for (i = 0; i < nprocs; i++) {
    sendcounts[i] = rows * n; // num elements to send
    if (i < remainder) {      // allocated remainder
      sendcounts[i] += n;
    }
    displs[i] = i * rows * n +
                (i < remainder ? i : remainder) *
                    n; // calculate displacement, taking into account remainder
  }


  MPI_Scatterv(matrix, sendcounts, displs, MPI_FLOAT, local_matrix,
               local_rows * n, MPI_FLOAT, 0, MPI_COMM_WORLD);


  for (i = 0; i < local_rows; i++) {
    local_result[i] = 0;
    for (j = 0; j < n; j++) {
      local_result[i] += local_matrix[i * n + j] * vector[j];
    }
  }


  float *result = NULL;
  result = (float *)malloc(n * sizeof(float));


  int *recvcounts = malloc(nprocs * sizeof(int));
  int *result_displs = malloc(nprocs * sizeof(int));


  for (i = 0; i < nprocs; i++) {
    recvcounts[i] = rows;
    if (i < remainder) {
      recvcounts[i] += 1;
    }
    result_displs[i] = i * rows + (i < remainder ? i : remainder);
  }


  MPI_Gatherv(local_result, local_rows, MPI_FLOAT, result, recvcounts,
              result_displs, MPI_FLOAT, 0, MPI_COMM_WORLD);


  free(local_matrix);
  free(local_result);
  free(sendcounts);
  free(displs);
  free(recvcounts);
  free(result_displs);


  return result;
}


void char_itter(char *input, int length){
  //Function which each process but 0 calls to read a portion of the input file and send the edge information to process 0
  int i = 0;
  int x;
  int y;
  int send[2];
  char node[1000] = "";
  char link[1000] = "";
  int index = 0;


  int first_val = 1;
  while(input[i] != '\0'){
    if (input[i] == '\n') { // newline so new entry
      if(first_val != 1){
        x = atoi(node) - 1; //-1 because of numbering vs array indexing
        y = atoi(link) - 1;
        send[0] = x;
        send[1] = y;


        //adj_matrix[j][i] = 1.0;        // connection found
        if(x != -1 && y != -1){
          MPI_Send(&send, 2, MPI_INT, 0,"", MPI_COMM_WORLD);
        }
      }
        index = 0;                        // reset
        first_val = 1;                    // reset
        memset(node, 0, strlen(node)); // reset strings
        memset(link, 0, strlen(link)); // reset strings


    } else if (input[i] == ' ') { // delimiter
      index = 0;
      first_val = 0;
    } else {
      if (first_val == 1) {
        node[index] = input[i];
      } else {
        link[index] = input[i];
      }
      index++;
    }
    i++;
  }
  send[0] = -1;
  send[1] = -1;
  MPI_Send(&send, 2, MPI_INT, 0,"", MPI_COMM_WORLD);
  //send "send" to 0
}


float **normalize(float **adj, int N, int ncols) {
  //normalizes a matrix with N rows and ncols columns by counting the number of ones in each column of the matrix and computing a new matrix of same size
  //then adjusts the values as described in the document.
  float *empty = (float *)malloc(N * ncols * sizeof(float));
  float **normed = floatalloc2d(N, ncols, empty);


  int count = 0;
  for (int j = 0; j < ncols; j++) {
    for (int i = 0; i < N; i++) {
      if (adj[i][j] == 1.0) {
        count++;
      }
      normed[i][j] = 0.0;
    }
    for (int i = 0; i < N; i++) {
      if (adj[i][j] == 1.0) {
        normed[i][j] = 1 / (float)count;
      } else {
        if (count == 0) {
          normed[i][j] = 1 / (float)N;
        }
      }
    }
    count = 0;
  }
  return normed;
}


float **googled(float **s, float alpha, int N, int ncols) {
  //applies google formula to each element of a matrix with N rows and ncols columns
  float cons = (1.00 - alpha) * (1.00 / N);
  float *empty = (float *)malloc(N * ncols * sizeof(float));
  float **google = floatalloc2d(N, ncols, empty);
  for (int i = 0; i < ncols; i++) {
    for (int j = 0; j < N; j++) {
      google[j][i] = (alpha * s[j][i]) + cons;
    }
  }
  return google;
}


float *pageRank(int rank, int nprocs, float *M, int iters, int N) {
  float *V = (float *)malloc(sizeof(float) * N);
  float *temp = (float *)malloc((sizeof(float) * N));
  for (int i = 0; i < N; i++) {
    // temp[i] = 0;
    V[i] = 1 / (float)N;
  }
  memcpy(temp, V, (N * sizeof(float))); // temp and V are equal
  int i = 0;
  while (i < iters) {
    memcpy(V, temp, (N * sizeof(float))); // update V
    // loop calculates new iteration of V, stored in temp
    // variable 'V' contains old V
    free(temp);
    temp = dot_prod(rank, nprocs, N, M, V);
    fflush(stdout);
    MPI_Bcast(temp, N, MPI_FLOAT, 0, MPI_COMM_WORLD); // init for all


    int *stat = (int *)malloc(sizeof(int));
    *stat = 0;
    if (rank == 0) {
      float diff_avg = 0.0;
      // calculate average difference between elements in current iteration and
      // last iteration
      for (int x = 0; x < N; x++) {
        diff_avg += fabs(temp[x] - V[x]);
      }
      diff_avg = diff_avg / (float)N;


      // if average difference is small enough, consider convergence reached
      if (diff_avg < 1e-9f) {
        printf("\nConvergence found in %d iterations\n", i + 1);
        *stat = 1;
        MPI_Bcast(stat, 1, MPI_INT, 0, MPI_COMM_WORLD); // init for all
        break;
      }
      MPI_Bcast(stat, 1, MPI_INT, 0, MPI_COMM_WORLD); // init for all
    } else {
      MPI_Bcast(stat, 1, MPI_INT, 0, MPI_COMM_WORLD); // init for all
      if (*stat == 1)
        break;
    }
    i++;
  }
  return V;
}


void floatfree2d(float **array) {
  free(array[0]);
  free(array);
  return;
}


void print_output(float *ranks, int print_size){
  for(int j = 0; j < print_size; j++){
    printf("Page: %d Rank: %.2f\n", j, ranks[j]);
  }
}


int main(int argc, char *argv[]) {
  int rank, nprocs;
  int N;
  char *filename;
  float alpha = 0.85;
  float **adj_matrix = NULL; //2d adjacency matrix
  float **normalized = NULL; //2d normalized matrix
  float **sub_adj_matrix = NULL; //subdivided adjacency matrix
  float **sub_goog_mat = NULL; //subdivided google matrix
  float **goog_mat; //google matrix
  float *sendptr = NULL; //pointer for scatter/gather
  float *empty = (float *)malloc(N * N * sizeof(float));
  float *empty2 = (float *)malloc(N * N * sizeof(float));


  double start_time, end_time, total_time;


  //Take input data
  if(argc < 3){
      printf("Error, missing arguments: required: {NodeSize}, {Filename}.\n");
      return 1;
  }


  N = atoi(argv[1]);
  filename = argv[2];


  //Initialize MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Status status;


  //Initialize MPI datatypes
  MPI_Datatype acol, acoltype, bcol, bcoltype;
  MPI_Type_vector(N, 1, N, MPI_FLOAT, &acol);
  MPI_Type_create_resized(acol, 0, 1 * sizeof(float), &acoltype);
  MPI_Type_commit(&acoltype);
  MPI_Type_vector(N, 1, N / nprocs, MPI_FLOAT, &bcol);
  MPI_Type_create_resized(bcol, 0, 1 * sizeof(float), &bcoltype);
  MPI_Type_commit(&bcoltype);


  fflush(stdout);


  //File reading
    int bufsize, buffchars;
    char *buffer;          /* Buffer for reading */
    MPI_Offset filesize;
    MPI_File myfile;    /* Shared file */


    //Open the file and get the size
    MPI_File_open (MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
       MPI_INFO_NULL, &myfile);
    MPI_File_get_size(myfile, &filesize);
    filesize = filesize/sizeof(char);
    if(rank != 0){
      bufsize = filesize/(nprocs - 1);
    } else {
      bufsize = 0;
    }
    //split the file into equal portions for each process
    buffer = (char *) malloc((bufsize+1)*sizeof(char));
    MPI_File_set_view(myfile, (rank - 1)*bufsize*sizeof(char), MPI_CHAR, MPI_CHAR,
          "native", MPI_INFO_NULL);
    MPI_File_read(myfile, buffer, bufsize, MPI_CHAR, &status);
    MPI_Get_count(&status, MPI_CHAR, &buffchars);
    /* Set terminating null char in the string */
    buffer[buffchars] = (char)0;


  if (rank != 0){
    //Every process but 0 calls this function to read their portion of the file which was split up
    char_itter(buffer, buffchars);


  } else {
    int finished_count = 0;
    int recv[2];
   
    adj_matrix = (float **)calloc(N ,sizeof(float *));
    for (int i = 0; i < N; i++){
      adj_matrix[i] = (float *)calloc(N, sizeof(float));
    }


    //float *calloced = (float *)calloc(N * N, sizeof(float));
    //adj_matrix = floatalloc2d(N, N, calloced);


    //Process 0 receives all of the edges each process reads from the file and send then adds the edge to the adjacency matrix
    while(finished_count < (nprocs - 1)){
      MPI_Recv(&recv, 2, MPI_INT, MPI_ANY_SOURCE, "", MPI_COMM_WORLD, &status);
      if(recv[0] == -1 && recv[1] == -1){
        finished_count++;
      } else {
        adj_matrix[recv[0]][recv[1]] = 1;
      }
    }
    sendptr = &(adj_matrix[0][0]);
  }
  MPI_Barrier(MPI_COMM_WORLD);


  //Start the time here as explained in the report
  start_time = MPI_Wtime();


  float *data = (float *)malloc(N * (N / nprocs) * sizeof(float));
  sub_adj_matrix = floatalloc2d(N, N / nprocs, data);
 
  //Split the adjacency matrix into even portions and scatter it across the proceses
  MPI_Scatter(sendptr, N / nprocs, acoltype, &(sub_adj_matrix[0][0]), N / nprocs, bcoltype,
              0, MPI_COMM_WORLD);
  free(adj_matrix);


  MPI_Barrier(MPI_COMM_WORLD);
  //each process runs the below function to normalize all of their columns
  normalized = normalize(sub_adj_matrix, N, N / nprocs);


  free(sub_adj_matrix);
 
  MPI_Barrier(MPI_COMM_WORLD);
  //each process runs the below function to apply the google formula to their columns
  sub_goog_mat = googled(normalized, alpha, N, N / nprocs);
  free(normalized);
  MPI_Barrier(MPI_COMM_WORLD);


  sendptr = &(sub_goog_mat[0][0]);


  goog_mat = floatalloc2d(N, N, empty);
 
  MPI_Allgather(sendptr, N / nprocs, bcoltype, &(goog_mat[0][0]), N / nprocs, acoltype, MPI_COMM_WORLD);
  //Gather each processes portion of the google matrix into goog_mat and share to all proceses
  free(sendptr);
  free(sub_goog_mat);
  MPI_Barrier(MPI_COMM_WORLD);
 
  float *oneD = (float *)calloc(N * (N / nprocs), sizeof(float));
  int start = rank * (N / nprocs);
  int count = 0;
  for (int i = start; i < (start + (N / nprocs)); i++){ //each process flattens their portion of the google matrix
    for (int j = 0; j < N; j++){
      if(count < (N * (N / nprocs))){
        oneD[count] = goog_mat[i][j];
      }
      count++;
    }
  }
  free(goog_mat);
  MPI_Barrier(MPI_COMM_WORLD);


  float *final_matrix = malloc(N * N * sizeof(float));
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allgather(oneD, N * (N / nprocs), MPI_FLOAT, final_matrix, N * (N / nprocs), MPI_FLOAT, MPI_COMM_WORLD);
  //Gatther each proceses portion of the flattened google matrix into final_matrix
  free(oneD);


  MPI_Barrier(MPI_COMM_WORLD);


  float *ranks = pageRank(rank, nprocs, final_matrix, 100, N); //Each process runs the pageRank function with a flattened google matrix
  free(final_matrix);


  if (rank == 0){ //Process 0 prints out computation time and the page rank scores
    end_time = MPI_Wtime();
    total_time = end_time - start_time;
    printf("Time elapsed: %lf seconds using %d processes.\n", total_time, p);
    print_output(ranks, N);
  }
  free(ranks);


  //Clean up MPI types
  MPI_Type_free(&acol);
  MPI_Type_free(&acoltype);
  MPI_Type_free(&bcol);
  MPI_Type_free(&bcoltype);


  MPI_Finalize(); //stop MPI
  return 0;
}

