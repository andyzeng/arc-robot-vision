% Use this for linux\mac.
if ispc
  mex mex_cbf_windows.cpp cbf_windows.cpp
else
  mex mex_cbf.cpp cbf.cpp
end