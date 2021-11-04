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

using Random
Random.seed!(1);

# Specify parameters.
N_tr = 50;
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

u, lf = build_svgp(ParameterHandling.value(θ_init))

# SVGP : Approximate posterior at f(z)

function build_svgp(θ)
    lf = build_latent_gp(θ)
    q = MvNormal(θ.qu.m, θ.qu.C + 1e-9I)
    fz = lf(θ.qu.z).fx
    return SVGP(fz, q), lf
end

function margin_to_expec(ps::AbstractVector{<:Normal})
    sqrt.(abs2.(mean.(ps)) .+ var.(ps))
end

function aug_optimize(u::SVGP, x_tr, y_tr; niter=3)
    K = ApproximateGPs._chol_cov(u.fz)
    q = u.q
    y = sign.(y_tr .- 0.5)
    θ = zeros(length(y_tr))
    for _ in 1:niter
        pf = marginals(posterior(u)(x_tr))
        @show c = margin_to_expec(pf)
        @. θ = tanh(c / 2) / (2c)
        Σ = Symmetric(inv(inv(K) + Diagonal(θ)))
        μ = Σ \ (y / 2 - K \ mean(u.fz))
        q = MvNormal(μ, Σ)
        u = SVGP(u.fz, q)
    end
    return u
end

u = aug_optimize(u, x_tr, y_tr)


function loss(θ)
    augsvgp, f = build_svgp(θ)
    fx = f(x_tr)
    svgp = Zygote.@ignore optimize(augsvgp, fx, y_tr)
    return -elbo(svgp, fx, y_tr)
end

# Optimise everything.
θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init);

# L-BFGS parameters chosen because they seems to work well empirically.
# You could also try with the defaults.
optimisation_result = optimize(
    loss ∘ unflatten,
    θ -> only(Zygote.gradient(loss ∘ unflatten, θ)),
    θ_flat_init + randn(length(θ_flat_init)),
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
