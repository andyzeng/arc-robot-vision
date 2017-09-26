
#include "cbf_windows.h"
#include "mex.h"

#define ARG_DEPTH 0
#define ARG_INTENSITY 1
#define ARG_NOISE 2
#define ARG_SIG_S 3
#define ARG_SIG_R 4

#define LOG_LEVEL 1

void validate_types(const mxArray* prhs[]) {
  if (mxGetClassID(prhs[ARG_DEPTH]) != mxUINT8_CLASS) {
    mexErrMsgTxt("Depth image must be of type uint8.");
  }
  
  if (mxGetClassID(prhs[ARG_INTENSITY]) != mxUINT8_CLASS) {
    mexErrMsgTxt("Intensity image must be of type uint8.");
  }
  
  if (mxGetClassID(prhs[ARG_NOISE]) != mxLOGICAL_CLASS) {
    mexErrMsgTxt("Noise image must be logical.");
  }
  
  if (mxGetClassID(prhs[ARG_SIG_S]) != mxDOUBLE_CLASS) {
    mexErrMsgTxt("SigmaS image must be double.");
  }
  
  if (mxGetClassID(prhs[ARG_SIG_R]) != mxDOUBLE_CLASS) {
    mexErrMsgTxt("SigmaR image must be double.");
  }
}

// Checks that all of the images are of the same size.
void validate_sizes(const mxArray* prhs[]) {
  if (mxGetNumberOfDimensions(prhs[ARG_DEPTH]) != 2) {
    mexErrMsgTxt("Depth image must be HxW");
  }
  
  if (mxGetNumberOfDimensions(prhs[ARG_INTENSITY]) != 2) {
    mexErrMsgTxt("Intensity image must be HxW");
  }
  
  if (mxGetNumberOfDimensions(prhs[ARG_NOISE]) != 2) {
    mexErrMsgTxt("Noise image must be HxW");
  }
  
  if (mxGetNumberOfDimensions(prhs[ARG_SIG_S]) != 2 || mxGetN(prhs[ARG_SIG_S]) != 1) {
    mexErrMsgTxt("SigamS must be Hx1");
  } else if (mxGetNumberOfElements(prhs[ARG_SIG_S]) != NUM_SCALES) {
    mexErrMsgTxt("SigmaS's length is fixed");  
  }
  
  if (mxGetNumberOfDimensions(prhs[ARG_SIG_R]) != 2 || mxGetN(prhs[ARG_SIG_R]) != 1) {
    mexErrMsgTxt("SigamR must be Hx1");
  } else if (mxGetNumberOfElements(prhs[ARG_SIG_R]) != NUM_SCALES) {
    mexErrMsgTxt("SigmaR's length is fixed");  
  }
  
  if (mxGetM(prhs[ARG_DEPTH]) != IMG_HEIGHT ||
      mxGetN(prhs[ARG_DEPTH]) != IMG_WIDTH) {
    mexErrMsgTxt("Depth image must be 480x640");
  }
  
  if (mxGetM(prhs[ARG_INTENSITY]) != IMG_HEIGHT ||
      mxGetN(prhs[ARG_INTENSITY]) != IMG_WIDTH) {
    mexErrMsgTxt("Intensity image must be 480x640");
  }
  
  if (mxGetM(prhs[ARG_NOISE]) != IMG_HEIGHT ||
      mxGetN(prhs[ARG_NOISE]) != IMG_WIDTH) {
    mexErrMsgTxt("Noise image must be 480x640");
  }
  
  int num_scales = mxGetM(prhs[ARG_SIG_S]);
  if (mxGetM(prhs[ARG_SIG_R]) != num_scales) {
    mexErrMsgTxt("SigmaS and SigmaR must be the same size (Sx1)");
  }
}

// Args:
//   depth - the HxW depth image (read in column major order).
//   intensity - the HxW intensity image (read in column major order).
//   noise_mask - the HxW logical noise mask. Values of 1 indicate that the 
//                corresponding depth value is missing or noisy.
//   sigma_s - Sx1 vector of sigmas.
//   sigma_r - Sx1 vector of range sigmas.
void mexFunction(int nlhs, mxArray* plhs[],
                 const int nrhs, const mxArray* prhs[]) {

  if (nrhs != 5) {
    mexErrMsgTxt("Usage: mex_cbf(depth, intensity, noise, sigmaS, sigmaR);");
  }

  validate_types(prhs);
  validate_sizes(prhs);

  uint8_t* depth = (uint8_t*) mxGetData(prhs[ARG_DEPTH]);
  uint8_t* intensity = (uint8_t*) mxGetData(prhs[ARG_INTENSITY]);
  bool* noise_mask = (bool*) mxGetData(prhs[ARG_NOISE]);
  double* sigma_s = (double*) mxGetData(prhs[ARG_SIG_S]);
  double* sigma_r = (double*) mxGetData(prhs[ARG_SIG_R]);

  mwSize ndim = 2;
  mwSize dims[] = {IMG_HEIGHT, IMG_WIDTH};
  plhs[0] = mxCreateNumericArray(ndim, &dims[0], mxUINT8_CLASS, mxREAL);
  uint8_t* result = (uint8_t*) mxGetData(plhs[0]);

  cbf::cbf(depth, intensity, noise_mask, result, sigma_s, sigma_r);  
}
