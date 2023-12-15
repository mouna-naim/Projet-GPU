# Projet_GPU
## Version AVX2: 
- La fonction **func_simd_avx2** est la fonction optimisée avec avx2 est qui marche le mieux.
- La fonction **naive_stencil_dunc1** est la fonction optimisée(Fusion des deux boucles sur x et y).


## Explication du code
  ## Version OpenMP
  - La fonction histogram_omp.cu est la version OpenMp du code. Pour compiler le code, vous tapez make -f Makefile sur le terminal et puis ./exec.

  ## Version CUDA
  - La fonction histogram_cuda_final.cu est la version cuda du code, pour compiler le code, vous tapez make -f Makefile1 sur le terminal et puis ./exec.
