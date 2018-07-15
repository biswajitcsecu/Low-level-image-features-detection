#pragma on
#define __cpluplus
#ifdef __INTELLISENSE__
void __syncthreads();
#endif
#include<cuda.h>
#include<cublas.h>
#include<cuda_device_runtime_api.h>
#include<device_launch_parameters.h>
#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include<device_functions.h>
#include<arrayfire.h>
#include<af/cuda.h>
#include<iostream>
#include<math.h>
#include<mkl.h>
#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

using namespace af;
using namespace std;

__device__ float debMultipliction(float ax, float by) {
float c;
c = ax*by;

return (c);
}
EXTERN_C __global__ void harrisCornerKernel(float *out, float *ina, float
*inb, float *inxx,
float *inyy, float *inxy, float *initr, float *inidet, int nx, int
ny)
{
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy*nx + ix;
if (ix < nx && iy < ny) {
inxx[idx] = debMultipliction(ina[idx], ina[idx]);
inyy[idx] = debMultipliction(inb[idx], inb[idx]);
inxy[idx] = debMultipliction(ina[idx], inb[idx]);
__syncthreads();
}if (ix < nx && iy < ny) {
initr[idx] = (inxx[idx] + inyy[idx]);
inidet[idx] = debMultipliction(inxx[idx], inyy[idx]) -
debMultipliction(inxy[idx], inyy[idx]);
inxy[idx] = debMultipliction(ina[idx], inb[idx]);
out[idx] = inidet[idx] - 0.04f*debMultipliction(initr[idx],
initr[idx]);
__syncthreads();
}
if (ix < nx && iy < ny) {
if (out[idx] > 1e5f) {
out[idx] = debMultipliction(out[idx], out[idx]);
__syncthreads();
}
__syncthreads();
}
}


static af::array harrisCornerMethod(af::array& gX, af::array& gY) {
af::array temp;
int h = gX.dims(0);
int w = gX.dims(1);
size_t nbyte = h*w*sizeof(f32);
// Get Arrayfire's internal CUDA stream
int af_id = af::getDevice();
cudaStream_t af_stream = afcu::getStream(af_id);
// allocate host memory
float* intA = (float *)mkl_malloc(nbyte, f32);
float* intB = (float *)mkl_malloc(nbyte, f32);
float* xx = (float *)mkl_malloc(nbyte, f32);
float* yy = (float *)mkl_malloc(nbyte, f32);
float* xy = (float *)mkl_malloc(nbyte, f32);
float* h_ref = (float *)mkl_malloc(nbyte, f32);
//data assignement
intA = gX.host<float>();
intB = gY.host<float>();
// allocate device memory
float *d_x, *d_y, *d_o, *d_xx, *d_yy, *d_xy, *d_itr, *d_idet;
cudaMalloc((float**)&d_x, nbyte);
cudaMalloc((float**)&d_y, nbyte);
cudaMalloc((float**)&d_xx, nbyte);
cudaMalloc((float**)&d_yy, nbyte);
cudaMalloc((float**)&d_xy, nbyte);
cudaMalloc((float**)&d_itr, nbyte);
cudaMalloc((float**)&d_idet, nbyte);cudaMalloc((float**)&d_o, nbyte);
// copy data from host to device
cudaMemcpy(d_x, intA, nbyte, cudaMemcpyHostToDevice);
cudaMemcpy(d_y, intB, nbyte, cudaMemcpyHostToDevice);
//kernel invock
dim3 block(32, 32);
dim3 grid((h + block.x - 1) / block.x, (w + block.y - 1) /
block.y);
harrisCornerKernel << < grid, block, 0, af_stream >> >
(d_o, d_x, d_y, d_xx, d_yy, d_xy, d_itr, d_idet, h, w);
// copy data from device to host
cudaMemcpy(h_ref, d_o, nbyte, cudaMemcpyDeviceToHost);
cudaThreadSynchronize();
cudaDeviceSynchronize();
temp = af::array(h, w, h_ref, afHost);
//memory free
cudaFree(d_y);
cudaFree(d_x);
cudaFree(d_xx);
cudaFree(d_yy);
cudaFree(d_xy);
cudaFree(d_itr);
cudaFree(d_idet);
cudaFree(d_o);
return(temp);
}
EXTERN_C __global__ void imageKernel(float *out, float *ina, float *inb,
int nx, int ny)
{
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy*nx + ix;
if (ix < nx && iy < ny) {
ina[idx] = 255.f - ina[idx];
inb[idx] = exp(ina[idx]);
out[idx] = (ina[idx])* inb[idx];
__syncthreads();
}
}
af::array test_nppiFilter(af::array &Grysrc)
{
int r = Grysrc.dims(0);
int c = Grysrc.dims(1);
size_t sz = r*c*sizeof(f32);
float* A = (float *)mkl_malloc(sz, f32);
float* h_ref = (float *)mkl_malloc(sz, f32);
//data assignementA = Grysrc.host<float>();
float *d_x, *d_y, *d_o;
cudaMalloc((float**)&d_x, sz);
cudaMalloc((float**)&d_y, sz);
cudaMalloc((float**)&d_o, sz);
cudaMemcpy(d_x, A, sz, cudaMemcpyHostToDevice);
//kernel invock
dim3 block(32, 32);
dim3 grid((r + block.x - 1) / block.x, (c + block.y - 1) /
block.y);
imageKernel << < grid, block, 0 >> >(d_o, d_x, d_y, r, c);
// copy data from device to host
cudaMemcpy(h_ref, d_o, sz, cudaMemcpyDeviceToHost);
cudaThreadSynchronize();
cudaDeviceSynchronize();
af::array temp = af::array(r, c, h_ref, afHost);
//memory free
cudaFree(d_x);
cudaFree(d_y);
cudaFree(d_o);
return(temp);
}
static void imageResDemo(bool console) {
// Load color image
af::array imsrcA;
imsrcA = loadImage("src.jpg", AF_RGB);
af::array Grysrc = af::colorSpace(imsrcA, AF_GRAY, AF_RGB);
//dimension of image
int rows = Grysrc.dims(0);
int cols = Grysrc.dims(1);
int chns = Grysrc.dims(2);
//creat new image
af::array nsrcA = moddims(Grysrc, rows, cols*chns);
af::array src = test_nppiFilter(Grysrc);
// pixel density region by Calculate image gradients
af::array ixA, iyA;
grad(ixA, iyA, nsrcA);
af::array hCMat = harrisCornerMethod(ixA, iyA);
af::array rimg = moddims(hCMat, rows, cols, chns);
af::array mask = constant(1, 3, 3);
af::array max_resp = dilate(rimg, mask);
rimg = (rimg == max_resp) * rimg;
// Gets host pointer to response data
float* h_corners = rimg.host<float>();
unsigned good_corners = 0;// display input images
af::array color_imgA = imsrcA / 255.f;
const int draw_len = 5;
for (int y = draw_len; y < color_imgA.dims(0) - draw_len; y++) {
for (int x = draw_len; x < color_imgA.dims(1) - draw_len;
x++) {
// Only draws crosshair if is a corner
if (h_corners[x * rimg.dims(0) + y] > 1e3f) {
// Draw horizontal line of (draw_len * 2 + 1)
pixels centered on the corner
// Set only the first channel to 1 (green lines)
color_imgA(y, seq(x - draw_len, x + draw_len), 0)
= 0.f;
color_imgA(y, seq(x - draw_len, x + draw_len), 1)
= 1.f;
color_imgA(y, seq(x - draw_len, x + draw_len), 2)
= 0.f;
// Draw vertical line of (draw_len * 2 + 1) pixels
centered on the corner
// Set only the first channel to 1 (green lines)
color_imgA(seq(y - draw_len, y + draw_len), x, 0)
= 0.f;
color_imgA(seq(y - draw_len, y + draw_len), x, 1)
= 1.f;
color_imgA(seq(y - draw_len, y + draw_len), x, 2)
= 0.f;
good_corners++;
}
}
}
//console
if (!console) {
af::Window wnd("CUDA Color Harris Corner Detection Demo");
wnd.setPos(50, 50);
while (!wnd.close()) {
wnd.grid(1, 2);
wnd(0, 0).image(imsrcA / 255.f, "Color Image");
wnd(0, 1).image(color_imgA, "Corner Col_Image");
wnd.show();
}
}
}
int main(int argc, char** argv) {
int device = argc > 1 ? atoi(argv[1]) : 0;
bool console = argc > 2 ? argv[2][0] == '_' : false;
int dev = 0;
cudaSetDevice(dev);
cudaDeviceProp devPro;
cudaGetDeviceProperties(&devPro, dev);
try {
//cuda
af::setDevice(device);af::info();
printf("Using Device %d: %s\n", dev, devPro.name);
// check if support mapped memory
if (!devPro.canMapHostMemory) {
printf("Device %d does not support mapping CPU host
memory!\n", dev);
cudaDeviceReset();
exit(EXIT_SUCCESS);
}
/*Function demo*/
imageResDemo(console);
}
catch (af::exception exp) {
fprintf(stderr, "%s\n", exp.what());
throw;
}
return(0);
}
