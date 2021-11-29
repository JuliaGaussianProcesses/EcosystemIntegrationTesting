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

using Plots
default(; legend=:outertopright, size=(700, 400))
include("polyagamma.jl")
using Random
Random.seed!(1);

# Specify parameters.
N_tr = 200;
x_tr = collect(range(-10.0, 10.0; length=N_tr));
θ_init = (
    gp = (
        σ² = positive(1.0),
        l = positive(3.0),
    ),
    qu = (
        z = x_tr,
        m = fixed(zeros(N_tr)),
        C = fixed(Matrix(Eye(N_tr))),
    ),
    jitter = fixed(1e-6),
);

# Specify functions to build LatentGP.
build_gp(θ) = GP(θ.σ² * with_lengthscale(SEKernel(), θ.l))

build_latent_gp(θ) = LatentGP(build_gp(θ.gp), BernoulliLikelihood(), θ.jitter)

# Generate synthetic data.

(f_tr, y_tr) = rand(build_latent_gp(ParameterHandling.value(θ_init))(x_tr));

# SVGP : Approximate posterior at f(z)

function build_svgp(θ)
    lf = build_latent_gp(θ)
    Zygote.@show lf.f.kernel
    q = MvNormal(θ.qu.m, θ.qu.C + 1e-9I)
    fz = lf(θ.qu.z).fx
    return SparseVariationalApproximation(Centered(), fz, q), lf
end

function marginals_to_aug_posterior!(qω::AbstractVector, ps::AbstractVector{<:Normal})
    map!(qω, ps) do p
        PolyaGamma(1, sqrt(abs2(mean(p)) + var(p)))
    end
end

function aug_optimize(u::SparseVariationalApproximation, x_tr, y_tr; niter=3)
    K = ApproximateGPs._chol_cov(u.fz)
    μ = mean(u.q)
    Σ = cov(u.q)
    y_sign = sign.(y_tr .- 0.5)
    qω = Vector{PolyaGamma{Int,Float64}}(undef, length(y_tr))
    for _ in 1:niter
        pf = marginals(posterior(u)(x_tr))
        marginals_to_aug_posterior!(qω, pf)
        Σ = Symmetric(inv(inv(K) + Diagonal(mean.(qω))))
        μ = Σ * (y_sign / 2 - K \ mean(u.fz))
    end
    return μ, Σ, qω
end

u, lf = build_svgp(ParameterHandling.value(θ_init))

m, S, qω = aug_optimize(u, x_tr, y_tr)
unew = SparseVariationalApproximation(Centered(), u.fz, MvNormal(m, S))
##
scatter(x_tr, y_tr, label="data", alpha=0.5)
plot!(x_tr, f_tr, label="")
plot!(posterior(unew)(x_tr))
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
    return  m / 2 - (abs2(m) + var(qf)) * mean(qω) / 2 -log(2)
end

function loss(θ)
    svgp, f = build_svgp(θ)
    fx = f(x_tr)
    μ, Σ, qω = Zygote.@ignore aug_optimize(svgp, x_tr, y_tr)
    augsvgp = SparseVariationalApproximation(Centered(), svgp.fz, MvNormal(μ, Σ))
    return -aug_elbo(augsvgp, fx, y_tr, qω)
end

# Optimise everything.
θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init);
unflatten(θ_flat_init)

(loss ∘ unflatten)(θ_flat_init)

Zygote.gradient(loss ∘ unflatten, θ_flat_init)

# L-BFGS parameters chosen because they seems to work well empirically.
# You could also try with the defaults.
optimisation_result = optimize(
    loss ∘ unflatten,
    θ -> only(Zygote.gradient(loss ∘ unflatten, θ)),
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
svgp_opt, lf_opt = build_svgp(θ_opt);

# Abstract these lines? Would be nice to write
# lf_post_opt = posterior(lf_opt, svgp_opt)
# or something.
post_opt = posterior(svgp_opt);
lf_post_opt = LatentGP(post_opt, lf_opt.lik, lf_opt.Σy);

f = [rand(lf_post_opt(x_pr)).f for _ in 1:20];
p = map(_f -> lf_opt.lik.invlink.(_f), f);

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
plot!(posterior(unew)(x_tr))