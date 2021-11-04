# Largely copied from Ti's example in ApproximateGPs.

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
x_tr = range(-10.0, 10.0; length=N_tr);
θ_init = (
    gp = (
        σ² = positive(1.0),
        l = positive(3.0),
    ),
    α = positive(1.0),
    qu = (
        z = fixed(x_tr),
        m = zeros(N_tr),
        C = positive_definite(Matrix(Eye(N_tr))),
    ),
    jitter = fixed(1e-6),
);

# Specify functions to build LatentGP.
build_gp(θ) = GP(θ.σ² * with_lengthscale(SEKernel(), θ.l))

build_latent_gp(θ) = LatentGP(build_gp(θ.gp), GammaLikelihood(θ.α), θ.jitter)

# Generate synthetic data.

(f_tr, y_tr) = rand(build_latent_gp(ParameterHandling.value(θ_init))(x_tr));

function build_svgp(θ)
    lf = build_latent_gp(θ)
    q = MvNormal(θ.qu.m, θ.qu.C + 1e-9I)
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
