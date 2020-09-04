/* A program to solve the hotplate problem in a parallel fashion using OpenSHMEM having a near 
 * linear speed up as PEs increase
 
 Author: Bukola Grace Omotoso
 Last Code Clean-up: 07/06/2019
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <shmem.h>

int counter = 0;
int num_rows;
int num_cols;
int start_row;
int stop_row;
float epsilon;
float** hotplate;
float** hotplateClone; 
static long pSync[SHMEM_BARRIER_SYNC_SIZE];
float global_max_diff = 0;

/* Build memory structure for hotplate*/
float** buildHotplate(int rows, int columns) {
    float** hotplate;
    hotplate = (float**) shmem_malloc(rows*sizeof(float*));
    for (int i = 0; i < rows; i++)
        hotplate[i] = (float*) shmem_malloc(columns*sizeof(float));
    return hotplate;
}

/*Build memory structure for hotplate clone*/
float** buildHotplateClone(int rows, int columns){
	float** hotplateClone;
	hotplateClone = (float**) shmem_malloc(rows*sizeof(float*));
	for (int i = 0; i < rows; i++)
		hotplateClone[i] = (float*) shmem_malloc(columns*sizeof(float));
	return hotplateClone;
}

void initializeHotPlate(int num_rows, int num_cols, float** hotplate, float** hotplateClone, int top_temp, int left_temp, int right_temp, int bottom_temp)    {
    int num_outer_grid = (2 * num_rows) + (2 * (num_cols - 2));
    float outer_grid_sum = (top_temp * (num_cols - 2)) + (left_temp * (num_rows - 1)) + (bottom_temp * num_cols) + (right_temp * (num_rows - 1));
    float initial_inner_val = outer_grid_sum / num_outer_grid;
  	  for (int row = 0; row < num_rows; row++) {
        for (int column = 0; column < num_cols; column++) {
            
            //top values override the top row except the edges
            if ((row == 0) & (column != 0 & column != num_cols - 1)) {
                hotplate[row][column] = top_temp;
                hotplateClone[row][column] = top_temp;
            }
            else if (column == 0 && (row != (num_rows-1))) {
                hotplate[row][column] = left_temp;
                hotplateClone[row][column] = left_temp;
            }
            else if (column == (num_cols - 1) && (row != (num_rows-1))) {
                hotplate[row][column] = right_temp;
                hotplateClone[row][column] = right_temp;
            }
            else if(row == (num_rows -1 )){
                hotplate[row][column] = bottom_temp;
                hotplateClone[row][column] = bottom_temp;
            }
            if ((row != 0) && (row != num_rows - 1) && (column != 0) && (column != num_cols - 1))
                hotplate[row][column] = initial_inner_val;
        }
    }
    
}

/*Get the maximum values from all threads*/
float max_max_diff(float arr[], int n)
{
    int i;
        float max = arr[0];
    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}

/* Swap hotplate and its clone*/
void swapHotplate(float *a, float *b) {
    
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

/* Get current time*/

double timestamp()
{
    struct timeval tval;
    
    gettimeofday( &tval, ( struct timezone * ) 0 );
    return ( tval.tv_sec + (tval.tv_usec / 1000000.0) );
}

/* Heating up hotplate*/

void generateHeat(int num_pes, int pe, int start_row, int stop_row, long pSync[]) {
    float* max_diffs = (float*)shmem_malloc(sizeof(float));
    float value = 0;
    float max_difference = 0;
    float previous_val = 0;
    float current_val = 0;
    float diff = 0;

    for (int row = start_row; row < stop_row; row++) {
        for (int column = 1; column < (num_cols - 1); column++) {
            previous_val = hotplate[row][column];
            current_val = ((hotplate[row - 1][column] + hotplate[row][column - 1] + hotplate[row + 1][column] + hotplate[row][column + 1]) / 4);         
	    diff = fabsf(previous_val - current_val);
            if (diff > max_difference){
                max_difference = diff;
            }
            hotplateClone[row][column] = current_val;
            shmem_float_put(&hotplateClone[row][column], &current_val, 1, 0);
        }
    }
     shmem_barrier_all();
    //Each PE sends its max_difference value to PE 0
    shmem_float_put(&max_diffs[pe], &max_difference,1, 0);
    if (num_pes > 1)
    	shmem_broadcast32(hotplateClone, hotplateClone, num_rows*num_cols, 0, 0, 0, num_pes, pSync); 
    //shmem_barrier_all();
    if (pe == 0){
	 global_max_diff = max_max_diff(max_diffs,num_pes);

}
}


int main(int argc, char const *argv[])
{
    shmem_init();
    int me = shmem_my_pe();
     int npes = shmem_n_pes();
    num_rows = atoi(argv[1]);
    num_cols = atoi(argv[2]);
    int top_temp = atoi(argv[3]);
    int left_temp = atoi(argv[4]);
    int right_temp = atoi(argv[5]);
    int bottom_temp = atoi(argv[6]);
    float epsilon = atof(argv[7]);
    for (int i = 0; i < SHMEM_BARRIER_SYNC_SIZE; i++) 
	pSync[i] = SHMEM_SYNC_VALUE;
        shmem_sync_all();
     
    hotplate =  buildHotplate(num_rows, num_cols);
    hotplateClone = buildHotplateClone(num_rows, num_cols);
    
    initializeHotPlate(num_rows, num_cols, hotplate, hotplateClone, top_temp, left_temp, right_temp, bottom_temp);
    int  row_per_pe = (num_rows + npes - 1)/npes;
    
    //divide the work among the available PEs
    
        start_row = me*row_per_pe;
        stop_row = (me + 1)*row_per_pe;
    
    /*Ensure that the first PE starts at row 1 and 
     * the last PE does not go past the end of a column*/
     
    if (me == 0){
	start_row = 1;
        printf("%-10s%10s\n", "Iteration", "Epsilon");
    }
    if (me == npes - 1)
    	stop_row = num_rows-1; 
    global_max_diff = epsilon + 1;
    double begin, end;
    
    begin = timestamp();
   while(global_max_diff > epsilon){
         generateHeat(npes, me, start_row, stop_row, pSync);
         swapHotplate((float*)&hotplate, (float*)&hotplateClone);
         if (((counter > 0 && (counter & (counter - 1)) == 0 )&& me == 0) || (global_max_diff < epsilon && me == 0))
             printf("%-10d%10.6f\n", counter, global_max_diff); 
	shmem_float_get(&global_max_diff, &global_max_diff, 1, 0);
         counter++;
       shmem_barrier_all();
    }
    end = timestamp();
    if (me == 0) 
    	printf("%s%5.2f\n","TOTAL TIME: ", (end-begin));
    shmem_finalize();
    return 0;
}
