# CUDA Parallelisation

Following a formulate for multivariate linear regression that takes in two variables only.
## Limitations

- Reads a maximum of 110k lines of data before throwing errors as shown below:
```
Unhandled exception at 0x00007FFCB7898288 (nvcuda64.dll) in dspc-cuda.exe: 0xC00000FD: Stack overflow (parameters: 0x0000000000000001, 0x000000DC4CE03000).
```
- Runs for approximately 1250 ms at 100k lines of data. Perhaps this time taken is relatively constant used for the launching of kernels.
