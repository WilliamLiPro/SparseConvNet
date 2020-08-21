Since the difference of win10 64bit and the Linux, several places has be modified for compile on win10:  
鉴于win10 64位系统与Linux之间的区别，为了让其在win10上成功运行，对代码做了如下修改：

(Reference: https://github.com/facebookresearch/SparseConvNet/issues/128)

1. In each Anaconda prompt, before "cl" is invoked, we need to run appropriate Visual Studio Developer command file to set up the environment variables. Example: "call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat". The exact batch file to execute depends on the architecture for which we want to compile etc.   
在编译安装之前，需要运行适当的编译器以设置环境变量，例如运行："C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"。具体运行的编译器取决于我们想要的编译架构。

2. I decided not to use develop.sh, because it leads to "Python: unknown command" error due to bash not being aware of Anaconda environment (??). The first line in develop.sh is for cleaning up -- which can be done manually. I chose to directly run the second line "python setup.py develop".   
由于bash指令无法在Anaconda环境下被识别，develop.sh不能使用。直接使用"python setup.py develop"指令安装。

3. There are  several occurrences of "or" (instead of "||") and "and" (instead of "&&") in Metadata.cpp, NetworkInNetwork.cpp etc.  
由于编译器无法识别“and”"or"，新的代码将原来的Metadata.cpp, NetworkInNetwork.cpp等文件的“and”"or"修改为“&&”“||”。

4. "tr1" was a temporary namespace that was in use when "functional" was still not part of the C++ standard. 
	- In "sparseconfig.h" under "sparseconvnet\SCN\Metadata\sparsehash\internal", made the following changes:
		- Changed "#define HASH_FUN_H <tr1/functional>" to "#define HASH_FUN_H <functional>"
		- Changed "#define HASH_NAMESPACE std::tr1" to "#define HASH_NAMESPACE std".  
对“sparseconvnet\SCN\Metadata\sparsehash\internal”下的 "sparseconfig.h" 进行如下修改：
		- 将"#define HASH_FUN_H <tr1/functional>" 改为 "#define HASH_FUN_H <functional>"
		- 将"#define HASH_NAMESPACE std::tr1" 改为"#define HASH_NAMESPACE std"

5. In “\SparseConvNet\sparseconvnet\SCN\Metadata\sparsehash\internal\hashtable-common.h”, changed the definition of SPARSEHASH_COMPILE_ASSERT(expr, msg) to static_assert(expr, "message")  
将原始的：
#define SPARSEHASH_COMPILE_ASSERT(expr, msg) \
	 __attribute__((unused)) typedef SparsehashCompileAssert<(bool(expr))> msg[bool(expr) ? 1 : -1]  
改为：
#define SPARSEHASH_COMPILE_ASSERT(expr, msg) static_assert(expr, "message")

6.  In“SparseConvNet\sparseconvnet\SCN\CUDA/SparseToDense.cpp”, Changed "std::array<long, Dimension + 2> sz" to "std::array<int64_t, Dimension + 2> sz"  
在“SparseConvNet\sparseconvnet\SCN\CUDA/SparseToDense.cpp”中，将"std::array<long, Dimension + 2> sz" 改为"std::array<int64_t, Dimension + 2> sz"

7. In "SparseConvNet\sparseconvnet\SCN\CUDA/BatchNormalization.cu", Changed "pow(_saveInvStd / nActive + eps, -0.5)" to "pow(double(_saveInvStd / nActive + eps), -0.5)". Otherwise, the calling signature happens to be pow(float, double) which does not correspond to the signature of any variant of "pow" function available on CUDA.  
 将“SparseConvNet\sparseconvnet\SCN\CUDA/BatchNormalization.cu”中的"pow(_saveInvStd / nActive + eps, -0.5)" 改为"pow(double(_saveInvStd / nActive + eps), -0.5)"，否则pow(float, double)将无法与CUDA中的"pow" 函数匹配。

8. As that poster said, code meant to be cross-platform should not be using "long". It would end up being 32 bit wide on 64-bit Windows machines while being 64 bit wide on 64-bit Linux machines.
	- Replaced all occurrences of "long" by "int64_t" and the mysterious link error went away.  
由于Linux和windows中的long定义长度不同，64-bit Windows上是32 bit wide，在64-bit Linux64 上是 64 bit wide，为了跨平台使用，需要将所有的"long" 数据声明改成"int64_t" 
