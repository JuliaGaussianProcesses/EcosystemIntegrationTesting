# Largely copied from Ti's example in ApproximateGPs.

using ApproximateGPs
using LinearAlgebra
using Distributions
using FillArrays
using LogExpFunctions: logistic, softplus, invsoftplus
using Optim
using ParameterHandling
using Zygote

using Plots
default(; legend=:outertopright, size=(700, 400))

using Random
Random.seed!(1);

# Specify parameters.
N_tr = 50;
x_tr = range(-10.0, 10.0; length=N_tr);
θ_init = (
    gp = (
        σ² = positive(1.0),
        l = positive(1.0),
    ),
    qu = (
        z = fixed(x_tr),
        m = zeros(N_tr),
        C = positive_definite(Matrix(Eye(N_tr))),
    ),
    jitter = fixed(1e-6),
);

# Specify functions to build LatentGP.
build_gp(θ) = GP(θ.σ² * with_lengthscale(SEKernel(), θ.l))

build_latent_gp(θ) = LatentGP(build_gp(θ.gp), BernoulliLikelihood(), θ.jitter)

# Generate synthetic data.

(f_tr, y_tr) = rand(build_latent_gp(ParameterHandling.value(θ_init))(x_tr));

function build_svgp(θ)
    lf = build_latent_gp(θ)
    q = MvNormal(θ.qu.m, θ.qu.C)
    fz = lf(θ.qu.z).fx
    return SVGP(fz, q), lf
end

function loss(θ)
    svgp, f = build_svgp(θ)
    fx = f(x_tr)
    return -elbo(svgp, fx, y_tr)
end

# Optimise everything.
θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init);

# L-BFGS parameters chosen because they seems to work well empirically.
# You could also try with the defaults.
result = optimize(
    loss ∘ unflatten,
    θ -> only(Zygote.gradient(loss ∘ unflatten, θ)),
    θ_flat_init,
    LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; iterations=100);
    inplace=false,
)




# function build_SVGP(params::NamedTuple)
#     kernel = params.k.var * (SqExponentialKernel() ∘ ScaleTransform(params.k.precision))
#     f = LatentGP(GP(kernel), lik, jitter)
#     q = MvNormal(params.m, params.A)
#     fz = f(params.z).fx
#     return SVGP(fz, q), f
# end

# function loss(params::NamedTuple)
#     svgp, f = build_SVGP(params)
#     fx = f(x)
#     return -elbo(svgp, fx, y)
# end;






# # Do optimisations
# new_parameters = ...

# # fixed some components and optimise further

# compute_objective ∘ unflatten













# x_plot = range(-13.0, 13.0; length=250)

# Xgrid = -4:0.1:29  # for visualization
# X = range(0, 23.5; length=48)  # training inputs
# f(x) = 3 * sin(10 + 0.6x) + sin(0.1x) - 1  # latent function
# fs = f.(X)  # latent function values at training inputs

# lik = BernoulliLikelihood()  # has logistic invlink by default
# # could use other invlink, e.g. normcdf(f) = cdf(Normal(), f)

# invlink = lik.invlink  # logistic function
# ps = invlink.(fs)  # probabilities at the training inputs
# Y = [rand(Bernoulli(p)) for p in ps]  # observations at the training inputs
# # could do this in one call as `Y = rand(lik(fs))`

# function plot_data()
#     plot(; xlims=extrema(Xgrid), xticks=0:6:24)
#     plot!(Xgrid, invlink ∘ f; label="true probabilities")
#     return scatter!(X, Y; label="observations", color=3)
# end

# plot_data()
