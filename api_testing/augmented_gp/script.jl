# Largely copied from Ti's example in ApproximateGPs.
using Pkg
Pkg.activate(@__DIR__)
using ApproximateGPs
using LinearAlgebra
using Distributions
using FillArrays
using Optim
using ParameterHandling
using Zygote
using ChainRulesCore
using PDMats
using Plots

using Random
Random.seed!(1);
ChainRulesCore.@opt_out ChainRulesCore.rrule(::Type{<:Matrix}, ::PDMats.PDMat)

# Specify parameters.
N_tr = 200;
x_tr = collect(range(-10.0, 10.0; length=N_tr));
N_z = 50;
z = collect(range(-10.0, 10.0; length=N_z))
θ_init = (
    gp = (
        σ² = positive(1.0),
        l = positive(3.0),
    ),
    z = z,
    jitter = fixed(1e-6),
);

function init_var_q(z, ::BernoulliLikelihood, x_tr)
    qω = Vector{PolyaGamma{Int,Float64}}(undef, length(x_tr))
    μ = zeros(length(z))
    Σ = Matrix{Float64}(I(length(z)))
    return μ, Σ, qω
end

μ, Σ, qω = init_var_q(z, BernoulliLikelihood(), x_tr)

# Specify functions to build LatentGP.
build_gp(θ) = GP(θ.σ² * with_lengthscale(SEKernel(), θ.l))

build_latent_gp(θ) = LatentGP(build_gp(θ.gp), BernoulliLikelihood(), θ.jitter)

# Generate synthetic data.

(f_tr, y_tr) = rand(build_latent_gp(ParameterHandling.value(θ_init))(x_tr));

# SVGP : Approximate posterior at f(z)

function build_fz_lf(θ)
    lf = build_latent_gp(θ)
    Zygote.@show lf.f.kernel
    fz = lf.f(θ.z)
    return fz, lf
end

function marginals_to_aug_posterior!(qω::AbstractVector, ps::AbstractVector{<:Normal})
    map!(qω, ps) do p
        PolyaGamma(1, sqrt(abs2(mean(p)) + var(p)))
    end
end

function aug_optimize(fz::AbstractGPs.FiniteGP, x_tr, y_tr, μ, Σ, qω; niter=3)
    K = ApproximateGPs._chol_cov(fz)
    @show κ = K \ cov(fz.f, fz.x, x_tr)
    y_sign = sign.(y_tr .- 0.5)
    @show first(μ)
    for _ in 1:niter
        pf = marginals(posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(μ, Σ)))(x_tr))
        marginals_to_aug_posterior!(qω, pf)
        Σ .= Symmetric(inv(inv(K) + project(Diagonal(mean.(qω)), κ)))
        μ .= Σ * (project(y_sign / 2, κ) - K \ mean(fz))
    end
    return μ, Σ, qω
end



function loss!(θ, μ, Σ, qω)
    fz, f = build_gpzf(θ)
    fx = f(x_tr)
    μ, Σ, qω = Zygote.@ignore aug_optimize(fz, x_tr, y_tr, μ, Σ, qω)
    augsvgp = SparseVariationalApproximation(Centered(), fz, MvNormal(μ, Σ))
    return -aug_elbo(augsvgp, fx, y_tr, qω)
end

project(x::AbstractVector, κ) = κ * x
project(x::AbstractMatrix, κ) = κ * x * κ'

##
scatter(x_tr, y_tr, label="data", alpha=0.5)
plot!(x_tr, f_tr, label="")
plot!(posterior(u)(x_tr))
##
function aug_elbo(
    sva::SparseVariationalApproximation,
    lfx::AbstractGPs.LatentFiniteGP,
    y::AbstractVector,
    aug_variables;
    num_data=length(y),
)
    @assert sva.fz.f === lfx.fx.f
    return _aug_elbo(sva, lfx.fx, y, lfx.lik, aug_variables, num_data)
end

function _aug_elbo(sva, fx, y, lik, aug_variables, num_data)
    @assert sva.fz.f === fx.f
    post = posterior(sva)
    q_f = marginals(post(fx.x))
    variational_exp = expected_aug_loglik(y, q_f, lik, aug_variables)

    n_batch = length(y)
    scale = num_data / n_batch
    return sum(variational_exp) * scale - Zygote.@ignore(kl_term(lik, aug_variables)) - ApproximateGPs.kl_term(sva, post)
    # return -ApproximateGPs.kl_term(sva, post)
end

function kl_term(::BernoulliLikelihood{<:LogisticLink}, aug_variables::AbstractVector{<:PolyaGamma})
    sum(aug_variables) do qω
        c = qω.c
        - abs2(c) * mean(qω) + 2log(cosh(c / 2))
    end
end

function expected_aug_loglik(y, q_f, lik, aug_variables)
    return map(y, q_f, aug_variables) do y, q, aug
        _expected_aug_loglik(lik, q, aug, y)
    end
end

function _expected_aug_loglik(::BernoulliLikelihood{<:LogisticLink}, qf, qω, y)
    m = mean(qf)
    return  m / 2 - (abs2(m) + var(qf)) * mean(qω) / 2 - log(2)
end



# Optimise everything.
θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init);
unflatten(θ_flat_init)

function hp_loss(μ, Σ, qω)
    return θ -> loss!(unflatten(θ), μ, Σ, qω)
end

hp_loss(μ, Σ, qω)(θ_flat_init)

Zygote.gradient(hp_loss(μ, Σ, qω), θ_flat_init)

# L-BFGS parameters chosen because they seems to work well empirically.
# You could also try with the defaults.
μ, Σ, qω = init_var_q(z, BernoulliLikelihood(), x_tr)

optimisation_result = optimize(
    hp_loss(μ, Σ, qω),
    θ -> only(Zygote.gradient(hp_loss(μ, Σ, qω), θ)),
    θ_flat_init,
    LBFGS(;
        alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
        linesearch=Optim.LineSearches.BackTracking(),
    ),
    Optim.Options(; iterations=1_000);
    inplace=false,
);


θ_opt = unflatten(optimisation_result.minimizer);

# 
x_pr = range(-15.0, 15.0; length=250);
fz, lf = build_gpzf(θ_opt);
svgp_opt = SparseVariationalApproximation(Centered(), fz, MvNormal(μ, Σ))
# Abstract these lines? Would be nice to write
# lf_post_opt = posterior(lf_opt, svgp_opt)
# or something.
post_opt = posterior(svgp_opt);
lf_post_opt = LatentGP(post_opt, lf.lik, lf.Σy);

f = [rand(lf_post_opt(x_pr)).f for _ in 1:20];
p = map(_f -> lf.lik.invlink.(_f), f);

let
    plt = plot()
    plot!(plt, x_pr, p; color=:blue, linealpha=0.2, label="")
    plot!(plt, x_pr, mean(p); color=:blue, linalpha=0.5, linewidth=2, label="")
    scatter!(plt, x_tr, y_tr; color=:red, label="")
end


## 
u, lf = build_svgp(ParameterHandling.value(θ_opt))

m, S, qω = aug_optimize(u, x_tr, y_tr)
unew = SparseVariationalApproximation(Centered(), u.fz, MvNormal(m, S))
##
scatter(x_tr, y_tr, label="data", alpha=0.5)
plot!(x_tr, f_tr, label="")
plot!(posterior(unew)(x_tr), label="Prediction on f")
scatter!(posterior(unew)(z), label="Inducing Points")



## Comparison With AGP.jl

using AugmentedGaussianProcesses

kernel_init = build_gp(ParameterHandling.value(θ_init.gp)).kernel
agp = AGP.SVGP(kernel_init, LogisticLikelihood(), AnalyticVI(), z; verbose=3)

train!(agp, x_tr, y_tr, 1000)

μ, sig = AGP.predict_f(agp, x_tr; cov=true)

kernel_init
agp[1].prior.kernel
kernel_opt = build_gp(ParameterHandling.value(θ_opt.gp)).kernel


plot!(x_tr, μ)