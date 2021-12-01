### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ faeb518e-651e-4829-87a9-a2fead7908dd
begin
	N_tr = 100
	x_tr = collect(range(-10.0, 10.0; length=N_tr))
	N_z = 50
	z = collect(range(-10.0, 10.0; length=N_z))
end;

# ╔═╡ 66953fb0-6903-462a-af33-f8db32230cfe
project(x::AbstractVector, κ) = κ * x

# ╔═╡ 11781066-4267-4843-b988-bf90b61574f8
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ 9bd5d65f-4c1b-4248-92df-463432bddafc
begin
	using Pkg
	Pkg.activate(@__DIR__)
	using ApproximateGPs
	using LinearAlgebra
	using Distributions
	using FillArrays
	using Optim
	using ParameterHandling
	using Zygote
	using PDMats
	using Plots
	default(; legend=:outertopright, size=(700, 400))
	ingredients("polyagamma.jl")
	using Random
	rng = Random.MersenneTwister(42);
end

# ╔═╡ ab903524-365d-4529-936f-98ef62b44544
θ_init = (;
    gp = (
        σ² = positive(1.0),
        l = positive(2.0),
    ),
    z,
    jitter = fixed(1e-6),
);

# ╔═╡ 3ccb0a88-63b2-45d8-b723-d613dfe28418
θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init);

# ╔═╡ e5cd8d0f-9efd-4dff-b97d-63bca5ee4971
function init_var_q(z, ::BernoulliLikelihood, x_tr)
    qω = Vector{PolyaGamma{Int,Float64}}(undef, length(x_tr))
    μ = zeros(length(z))
    Σ = Matrix{Float64}(I(length(z)))
    return μ, Σ, qω
end;

# ╔═╡ 6068ea57-13b0-4a54-a79a-3be978379f76
# Initialize the posterior MvNormal parameters and the posterior distributions of ω
μ, Σ, qω = init_var_q(z, BernoulliLikelihood(), x_tr);

# ╔═╡ 609c8f65-c320-460b-ab81-547f35f74248
build_gp(θ) = GP(θ.σ² * with_lengthscale(SEKernel(), θ.l))

# ╔═╡ 644d6f7d-16a3-4ae6-9463-bc2bacde4485
build_latent_gp(θ) = LatentGP(build_gp(θ.gp), BernoulliLikelihood(), θ.jitter)

# ╔═╡ 1bd89171-b79a-4dfc-bcef-42610f46e366
(f_tr, y_tr) = rand(rng, build_latent_gp(ParameterHandling.value(θ_init))(x_tr));

# ╔═╡ 6e80e323-456f-467a-b1d2-130317f67db2
begin
	p_data = scatter(x_tr, y_tr, label="data", title="Data", xlabel="x")
	plot!(x_tr, f_tr, lw=3.0, label="latent gp")
end

# ╔═╡ 16271c13-8091-4192-895f-3f8afc26fc90
function build_fz_lf(θ)
    lf = build_latent_gp(θ)
    fz = lf(θ.z).fx
    return fz, lf
end

# ╔═╡ f3150669-dbe1-442d-b336-d01cab2babd3
function marginals_to_aug_posterior!(qω::AbstractVector, ps::AbstractVector{<:Normal})
    map!(qω, ps) do p
        PolyaGamma(1, sqrt(abs2(mean(p)) + var(p)))
    end
end

# ╔═╡ 55e30726-d959-4063-8f12-e3cda968f974
project(x::AbstractMatrix, κ) = κ * x * κ'

# ╔═╡ 1b1a79db-af79-4665-a96a-deda8c6633b6
# Update the variational parameters in place
function aug_optimize!(fz::AbstractGPs.FiniteGP, x_tr, y_tr, μ, Σ, qω; niter=10)
    K = ApproximateGPs._chol_cov(fz)
    κ = K \ cov(fz.f, fz.x, x_tr)
    y_sign = sign.(y_tr .- 0.5)
    for _ in 1:niter
        pf = marginals(posterior(SparseVariationalApproximation(Centered(), fz, MvNormal(μ, Σ)))(x_tr))
        marginals_to_aug_posterior!(qω, pf)
        Σ .= Symmetric(inv(inv(K) + project(Diagonal(mean.(qω)), κ)))
        μ .= Σ * (project(y_sign / 2, κ) - K \ mean(fz))
    end
    return μ, Σ, qω
end


# ╔═╡ 5493925c-1cfb-4a87-9468-3379779e56c7
begin
	fz_init, lf_init = build_fz_lf(ParameterHandling.value(θ_init))
	aug_optimize!(fz_init, x_tr, y_tr, μ, Σ, qω)
	u_init = SparseVariationalApproximation(Centered(), fz_init, MvNormal(μ, Σ))
	plot!(p_data, posterior(u_init)(x_tr), label="Initial posterior prediction")
end

# ╔═╡ 2b5e129e-4b80-4b1a-807b-f85004729d48
function kl_term(::BernoulliLikelihood{<:LogisticLink}, aug_variables::AbstractVector{<:PolyaGamma})
    sum(aug_variables) do qω
        c = qω.c
        - abs2(c) * mean(qω) + 2log(cosh(c / 2))
    end
end

# ╔═╡ 71154a8c-6bf2-4b76-b1ca-2f67b90432b6
function _expected_aug_loglik(::BernoulliLikelihood{<:LogisticLink}, qf, qω, y)
    m = mean(qf)
    return  m / 2 - (abs2(m) + var(qf)) * mean(qω) / 2 - log(2)
end

# ╔═╡ 5069065f-2161-42cf-8591-0da4d240f2ed
function expected_aug_loglik(y, q_f, lik, aug_variables)
    return map(y, q_f, aug_variables) do y, q, aug
        _expected_aug_loglik(lik, q, aug, y)
    end
end

# ╔═╡ 15f5b1be-11d0-4b35-bc0f-da9f57d93b6c
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

# ╔═╡ 3969259c-9c4f-4497-8f0c-ef658286a535
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

# ╔═╡ 0b79c288-b640-4ebe-afdf-f85bb0336240
# Compute the loss given theta and the variational parameters while updating mu and Sigma
function loss!(θ, μ, Σ, qω)
    fz, lf = build_fz_lf(θ)
    fx = lf(x_tr)
    Zygote.@ignore aug_optimize!(fz, x_tr, y_tr, μ, Σ, qω)
    augsvgp = SparseVariationalApproximation(Centered(), fz, MvNormal(μ, Σ))
    return -aug_elbo(augsvgp, fx, y_tr, qω)
end

# ╔═╡ 1a2dbd9e-4f92-45b6-b05d-ef845dba79f8
function hp_loss(μ, Σ, qω)
    return θ -> loss!(unflatten(θ), μ, Σ, qω)
end

# ╔═╡ 9024c702-8fb5-43fc-8300-29dd50af699d
hp_loss(μ, Σ, qω)(θ_flat_init)

# ╔═╡ 81df4519-f209-4393-88fc-f369eeab9318
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

# ╔═╡ b10fd344-a76a-4cc4-957a-adf2befb7144
θ_opt = ParameterHandling.value(unflatten(optimisation_result.minimizer))

# ╔═╡ 8fd6b335-0b90-4534-9785-1ecc8ea698cd
begin
	fz_opt, lf_opt = build_fz_lf(θ_opt)
	aug_optimize!(fz_init, x_tr, y_tr, μ, Σ, qω)
	u_opt = SparseVariationalApproximation(Centered(), fz_opt, MvNormal(μ, Σ))
	plot!(p_data, posterior(u_opt)(x_tr), label="Final posterior prediction")
end

# ╔═╡ Cell order:
# ╠═9bd5d65f-4c1b-4248-92df-463432bddafc
# ╠═faeb518e-651e-4829-87a9-a2fead7908dd
# ╠═ab903524-365d-4529-936f-98ef62b44544
# ╠═1bd89171-b79a-4dfc-bcef-42610f46e366
# ╠═6e80e323-456f-467a-b1d2-130317f67db2
# ╠═6068ea57-13b0-4a54-a79a-3be978379f76
# ╠═5493925c-1cfb-4a87-9468-3379779e56c7
# ╠═3ccb0a88-63b2-45d8-b723-d613dfe28418
# ╠═1a2dbd9e-4f92-45b6-b05d-ef845dba79f8
# ╠═9024c702-8fb5-43fc-8300-29dd50af699d
# ╠═81df4519-f209-4393-88fc-f369eeab9318
# ╠═b10fd344-a76a-4cc4-957a-adf2befb7144
# ╠═8fd6b335-0b90-4534-9785-1ecc8ea698cd
# ╠═e5cd8d0f-9efd-4dff-b97d-63bca5ee4971
# ╠═609c8f65-c320-460b-ab81-547f35f74248
# ╠═644d6f7d-16a3-4ae6-9463-bc2bacde4485
# ╠═16271c13-8091-4192-895f-3f8afc26fc90
# ╠═f3150669-dbe1-442d-b336-d01cab2babd3
# ╠═1b1a79db-af79-4665-a96a-deda8c6633b6
# ╠═66953fb0-6903-462a-af33-f8db32230cfe
# ╠═55e30726-d959-4063-8f12-e3cda968f974
# ╠═0b79c288-b640-4ebe-afdf-f85bb0336240
# ╠═3969259c-9c4f-4497-8f0c-ef658286a535
# ╠═15f5b1be-11d0-4b35-bc0f-da9f57d93b6c
# ╠═2b5e129e-4b80-4b1a-807b-f85004729d48
# ╠═5069065f-2161-42cf-8591-0da4d240f2ed
# ╠═71154a8c-6bf2-4b76-b1ca-2f67b90432b6
# ╟─11781066-4267-4843-b988-bf90b61574f8
