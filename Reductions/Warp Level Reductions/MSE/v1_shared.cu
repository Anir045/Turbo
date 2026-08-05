__global__ void mse(float *y_pred, float *y_actual, int N, float *res){
    __shared__ float y[BLOCK_DIM];
    int bx = blockIdx.x; int tx = threadIdx.x;
    int tid = bx * blockDim.x + tx;
    int strides = blockDim.x * gridDim.x;
    float diff = 0.0f;
    float sum =  0.0f;

    for(int i=tid;i<N;i+=4*strides){
        if(i<N){
            diff = y_actual[i] - y_pred[i];
            sum += diff*diff;
        } if(i+1*strides<N){
            diff = y_actual[i+1*strides] - y_pred[i+1*strides];
            sum += diff*diff;
        } if(i+2*strides<N){
            diff = y_aactual[i+2*strides] - y_pred[i+2*strides];
            sum += diff*diff;
        } if(i+3*strides<N){
            diff = y_aactual[i+3*strides] - y_pred[i+3*strides];
            sum += diff*diff;
        }
    }
    y[tx] = sum;
    __syncthreads();
    for(int j=blockDim.x/2;j>0;j/=2){
        if(tx<j){
            y[tx] += y[tx+j];
        }
        __syncthreads();
    }
    if(tx==0){
        atomicAdd(res, y[0]);
    }
}