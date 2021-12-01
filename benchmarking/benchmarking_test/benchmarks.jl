using BenchmarkTools
using KernelFunctions

X = rand(10, 20)
Xc = ColVecs(X)
Xr = RowVecs(X)
Xv = collect.(eachcol(X))
Y = rand(10, 20)
Yc = ColVecs(Y)
Yr = RowVecs(Y)
Yv = collect.(eachcol(Y))
SUITE = BenchmarkGroup()

kernels = Dict(
    "SqExponential" => SqExponentialKernel(),
    "Exponential" => ExponentialKernel(),
)

inputtypes = Dict(
    "ColVecs" => (Xc, Yc),
    "RowVecs" => (Xr, Yr),
    "Vecs" => (Xv, Yv),
)

functions = Dict(
    "kernelmatrixX" => (kernel, X, Y) -> kernelmatrix(kernel, X),
    "kernelmatrixXY" => (kernel, X, Y) -> kernelmatrix(kernel, X, Y),
    "kernelmatrix_diagX" => (kernel, X, Y) -> kernelmatrix_diag(kernel, X),
    "kernelmatrix_diagXY" => (kernel, X, Y) -> kernelmatrix_diag(kernel, X, Y)
)

for (kname, kernel) in kernels
    SUITE[kname] = sk = BenchmarkGroup()
    for (inputname, (X, Y)) in inputtypes
        sk[inputname] = si = BenchmarkGroup()
        for (fname, f) in functions
            si[fname] = @benchmarkable $f($kernel, $X, $Y)
        end
    end
end

# Uncomment the following to run benchmark locally

# tune!(SUITE)

# results = run(SUITE, verbose=true)