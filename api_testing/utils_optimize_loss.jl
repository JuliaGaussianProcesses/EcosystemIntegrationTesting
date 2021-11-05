using Optim, Zygote, ParameterHandling

# L-BFGS parameters chosen because they seem to work well empirically.
# You could also try with the defaults.
default_optimizer = LBFGS(;
    alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),
)

function optimize_loss(loss, θ_init; optimizer=default_optimizer, maxiter=1_000)
    options = Optim.Options(; iterations=maxiter, show_trace=true)

    θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init)
    loss_packed = loss ∘ unflatten 

    # https://julianlsolvers.github.io/Optim.jl/stable/#user/tipsandtricks/#avoid-repeating-computations
    function fg!(F, G, x)
        if F != nothing && G != nothing
            val, grad = Zygote.withgradient(loss_packed, x)
            G .= only(grad)
            return val
        elseif G != nothing
            grad = Zygote.gradient(loss_packed, x)
            G .= only(grad)
            return nothing
        elseif F != nothing
            return loss_packed(x)
          end
    end

    result = optimize(
        Optim.only_fg!(fg!),
        θ_flat_init,
        optimizer,
        options;
        inplace=false,
    )

    return unflatten(result.minimizer), result
end
