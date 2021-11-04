## ISSUES:
# - Why does CategoricalLikelihood add a zero to the input?
# - No IndependentMOKernel for different kernels on each output
# - LFGP sampler doesn't work with multi-output
# - elbo/expected_loglik needs a more general approach for multi-output GPs

using ApproximateGPs
using GPLikelihoods
using Distributions
using StatsFuns
using ParameterHandling
using LinearAlgebra

# The CategoricalLikelihood in GPLikelihoods currently takes C-1 inputs for C
# classes and defines the last one as 0 - technically this is fine but seems a bit unnatural?
# Also, when using a different kernel per class I don't think this works?
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

lgp = LatentGP(gp, lik, jitter)
lgp_fx = LatentGP(gp, lik, jitter)(X_mo)

# # TODO: rand doesn't work - f is a flat vector, needs to be reshaped before
# lik(f) is called This is essentially the same problem as the ELBO below but
# for LatentGPs. Should the dimensionality be defined by the likelihood (would
# need to specify num_classes in CategoricalLikelihood for example)
f_broken, y_broken = rand(lfgp)

f = rand(gp(X_mo, jitter))

rowvecs_y(f, out_dim) = RowVecs(reshape(f, :, out_dim))

rf = rowvecs_y(f, C)
y = rand(lik(rf))

onehot(y) = (sort(unique(y))' .== y)
# y = onehot(y)

M = N  # Number of inducing points

Z = copy(X)[1:M]
Z_mo = KernelFunctions.MOInputIsotopicByOutputs(Z, C)
fz = lgp(Z_mo).fx

q = MvNormal(zeros(M * C), Matrix{Float64}(I, M * C, M * C))

svgp = SVGP(fz, q)

post = posterior(svgp)

#!! A nasty hack to get the ELBO to work.
# The essential problem is that q_f is
# passed as a flat vector, but needs to be reshaped according to the number of
# classes. This approach works if the marginals q_f are independent for each
# class, but in general the marginals might need to be MvNormal.

# Where should this be fixed? Ideally a general solution that lets `marginals`
# work with MOGPs?
function expected_loglik_multiclass(mc::MonteCarlo, y::AbstractVector, q_f::AbstractVector{<:Normal}, lik; num_classes=C)
    q_f_ = reshape(q_f, :, num_classes)
    f_μ = mean.(q_f_)
    fs = f_μ .+ std.(q_f_) .* randn(eltype(f_μ), tuple(size(q_f_)..., mc.n_samples))
    p_ys = mapslices(lik, fs, dims=2)
    lls = loglikelihood.(p_ys, y)
    return sum(lls) / mc.n_samples
end


function ApproximateGPs.expected_loglik(mc::MonteCarlo, y::AbstractVector, q_f::AbstractVector{<:Normal}, lik::CustomCategorical)
    return expected_loglik_multiclass(mc, y, q_f, lik; num_classes=C)
end

elbo(svgp, lgp_fx, py; quadrature=MonteCarlo())
