#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "./ppmFile.c" 

#define MAX_VALUE 1000
#define MAX_WIDTH 1680
#define MAX_HEIGHT 1050

struct Point{
    float r,g,b;
    float x,y;
};

cudaError_t cuda_ret;

__global__ void kernel( Point *p, unsigned char *ptr, int sites ) {
    // Map from blockIdx to pixel position
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    int offset = x + (y * blockDim.x*gridDim.x);

    // Now find the closest site and color the current pixel accordingly
    // Hint: In a for loop, measure the Euclidean Distance between the current
    //       pixel and each site. Find the shortest.


    int minDistance = sqrt(pow(MAX_WIDTH,2) + pow(MAX_HEIGHT,2));//LONG_MAX;
    int closest_pt_idx = -1;
    //find closest point
    for(unsigned int i = 0; i < sites; i++){
      int distance = sqrt(pow(p[i].x - x, 2) + pow(p[i].y - y, 2));
      if(distance < minDistance){
        minDistance = distance;
        closest_pt_idx = i;
      } 
    }
    //set color
    ptr[offset*3+0] = p[closest_pt_idx].r;
    ptr[offset*3+1] = p[closest_pt_idx].g;
    ptr[offset*3+2] = p[closest_pt_idx].b;


}

int getDivisor(int numerator)
{
    int divisor = 31;
    while((numerator % divisor) != 0) {divisor--;}
    return divisor;
}

int main(int argc, char **argv) {

    if(argc < 4) {
        printf("usage: voronoi <width> <height> <sites>");
        exit(0);
    }

    // 1. Obtain width, height of the Voronoi Diagram image and the number of sites from command line ------
    // Ensure they are within the pre-set max caps 

    int w = atoi(argv[1]), h = atoi(argv[2]), points = atoi(argv[3]);
    w = (w <= MAX_WIDTH)? w : MAX_WIDTH;
    h = (h <= MAX_HEIGHT)? h : MAX_HEIGHT;
    points = (points <= MAX_VALUE)? points : MAX_VALUE;

    // 2. Allocate device memory for the Voronoi Diagram image ----------------------------------------------
    // Use "Julia Set" for reference
    unsigned char *d_copy;
    cuda_ret = cudaMalloc((void** )&d_copy, sizeof(unsigned char*) * (w*h*3));
    if(cuda_ret != cudaSuccess) fprintf(stderr, "%s\n", "Unable to allocate device memory (Voronoi Diagram Image)");

    // 3. Sites --------------------------------------------------------------------------------------------
    // 3.1. Allocate device memory for the sites 
    // The type of each site is a structure named "Point" defined at the top of this program
    struct Point *d_Sites;
    cuda_ret = cudaMalloc((void**) &d_Sites, sizeof(struct Point) * points);
    if(cuda_ret != cudaSuccess) fprintf(stderr, "%s\n", "Unable to allocate device memory (Sites)");

    // 3.2. Allocate host memory for the the sites. 
    // Then for each site, 
    //  - assign a random value say rand()%255+50 to its r, g, and b elemment respectively
    //  - assign a random value say rand()%w to its x, where w is the width of the Voronoi Diagram image
    //  - assign a random value say rand()%h to its y, where h is the height of the Voronoi Diagram image
    struct Point *h_Sites = (struct Point*)malloc(sizeof(struct Point) * points);
    for(unsigned int i = 0; i < points; i++){

      h_Sites[i].r = rand()%255 + 50;
      h_Sites[i].g = rand()%255 + 50;
      h_Sites[i].b = rand()%255 + 50;

      h_Sites[i].x = rand()%w;
      h_Sites[i].y = rand()%h;
    }

    // 3.3. Copy all sites from host memory to device memory
    printf("Copying all sites from host to device...");
    cuda_ret = cudaMemcpy(d_Sites, h_Sites, sizeof(struct Point) * points, cudaMemcpyHostToDevice);

    // Launch kernel ----------------------------------------------------------------------------------------
    // The pre-defined function "getDivisor" comes convenient. 
    // When it's called with w, getDivisor(w), it generates the largest number (<=31) that divides w 
    // When it's called with h, getDivisor(h), it generates the largest number (<=31) that divides h
    // These two results become demisions of a block 
    int wGrid = getDivisor(w);
    int hGrid = getDivisor(h);

    dim3 dimGrid(w/wGrid, h/hGrid);
    dim3 dimBlock(wGrid, hGrid);

    printf("\ndimGrid: (%d, %d)", w/wGrid, h/hGrid);
    printf("\ndimBlock: (%d, %d)", wGrid, hGrid);
    printf("\nResolution: %dx%d", w, h);
    printf("\nSites: %d\n\n", points);

    // Replace the following line with your code ...
    kernel<<<dimGrid,dimBlock>>>(d_Sites, d_copy, points);



    Image *outImage;
    outImage = ImageCreate(w,h);
    ImageClear(outImage,0,0,0);
    
    // Copy device variables to host ----------------------------------------
    // Use "Julia Set" for reference
    cuda_ret = cudaMemcpy(outImage -> data, d_copy, w*h*3, cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) fprintf(stderr, "%s\n", "Unable to copy device variables to host (Sites)");


    // Convert image to the ppm formt and free the host memory.----------------------------------------
    // Use "Julia Set" for reference
    const char* outFile = "out.ppm";
    ImageWrite(outImage, outFile);

    // Free device memory
    free(h_Sites);
    cudaFree(d_Sites); cudaFree(d_copy);
    free(outImage->data);
    
}

