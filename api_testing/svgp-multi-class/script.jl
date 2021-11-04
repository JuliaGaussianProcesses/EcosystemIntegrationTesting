## ISSUES:
# - Why does CategoricalLikelihood add a zero to the input?
# - No IndependentMOKernel for different kernels on each output
# - LFGP sampler doesn't work with multi-output
# - elbo/expected_loglik doesn't work with categorical likelihood

using ApproximateGPs
using GPLikelihoods
using Distributions
using StatsFuns
using ParameterHandling
using LinearAlgebra

struct CustomCategorical{Tl<:GPLikelihoods.AbstractLink} <: GPLikelihoods.AbstractLikelihood
    invlink::Tl
end

CustomCategorical(l=softmax) = CustomCategorical(Link(l))

(l::CustomCategorical)(f::AbstractVector{<:Real}) = Categorical(l.invlink(f))

function (l::CustomCategorical)(fs::AbstractVector)
    return Product(Categorical.(l.invlink.(fs)))
end


C = 3
N = 100

X = rand(N)
X_mo = KernelFunctions.MOInputIsotopicByOutputs(X, C)

lengthscale = 0.1

k = with_lengthscale(SEKernel(), lengthscale)
mo_kernel = IndependentMOKernel(k)

jitter = 1e-12

lik = CustomCategorical()

gp = GP(mo_kernel)

# # TODO: doesn't work - f is a flat vector, needs to be reshaped before lik(f) is called
lgp = LatentGP(gp, lik, jitter)
lgp_fx = LatentGP(gp, lik, jitter)(X_mo)
# rand(lfgp)

f = rand(gp(X_mo, jitter))

rowvecs_y(f, out_dim) = RowVecs(reshape(f, :, out_dim))
rf = rowvecs_y(f, C)


py = rand(lik(rf))

onehot(y) = (sort(unique(y))' .== y)
y = onehot(py)

M = N  # Number of inducing points

Z = copy(X)[1:M]
Z_mo = KernelFunctions.MOInputIsotopicByOutputs(Z, C)
fz = lgp(Z_mo).fx

q = MvNormal(zeros(M * C), Matrix{Float64}(I, M * C, M * C))

svgp = SVGP(fz, q)

post = posterior(svgp)  # Appears to work


# TODO: This is broken because we need to know the number or classes (in general
# the input dims for the likelihood) to compute the `expected_loglik`.
# Should this be defined in LatentGP or the Likelihood?
elbo(svgp, lgp_fx, vec(y); quadrature=MonteCarlo())
