FROM nightseas/cuda-torch
RUN luarocks install hdf5
RUN luarocks install inn
