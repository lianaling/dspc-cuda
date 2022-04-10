# CUDA Parallelisation

## Limitations

- Reads maximum of 110k lines before throwing errors as shown below:
```
Unhandled exception at 0x00007FFCB7898288 (nvcuda64.dll) in dspc-cuda.exe: 0xC00000FD: Stack overflow (parameters: 0x0000000000000001, 0x000000DC4CE03000).
```
