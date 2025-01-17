using OceanBioME, Oceananigans, Test, JLD2

using OceanBioME.Sediments: SimpleMultiG, InstantRemineralisation
using Oceananigans.Units

using OceanBioME.Sediments: sediment_tracers, sediment_fields
using Oceananigans: Field
using Oceananigans.Fields: TracerFields

using Oceananigans.Operators: volume, Azᶠᶜᶜ

function intercept_tendencies!(model, intercepted_tendencies)
    for tracer in keys(model.tracers)
        copyto!(intercepted_tendencies[tracer], model.timestepper.Gⁿ[tracer])
    end
end

function set_defaults!(sediment::SimpleMultiG)
    set!(sediment.fields.N_fast, 0.0230)
    set!(sediment.fields.N_slow, 0.0807)

    set!(sediment.fields.C_fast, 0.5893)
    set!(sediment.fields.C_slow, 0.1677)
end 

set_defaults!(::InstantRemineralisation) = nothing

function set_defaults!(::LOBSTER, model)
    set!(model, P = 0.4686, Z = 0.5363, 
                NO₃ = 2.3103, NH₄ = 0.0010, 
                DIC = 2106.9, Alk = 2408.9, 
                O₂ = 258.92, 
                DOC = 5.3390, DON = 0.8115,
                sPON = 0.2299, sPOC = 1.5080,
                bPON = 0.0103, bPOC = 0.0781)
end

set_defaults!(::NutrientPhytoplanktonZooplanktonDetritus, model) =  set!(model, N = 2.3, P = 0.4, Z = 0.5, D = 0.2)

total_nitrogen(sed::SimpleMultiG) = sum(sed.fields.N_fast) + 
                                    sum(sed.fields.N_slow) + 
                                    sum(sed.fields.N_ref)

total_nitrogen(sed::InstantRemineralisation) = sum(sed.fields.N_storage)

total_nitrogen(::LOBSTER, model) = sum(model.tracers.NO₃) + sum(model.tracers.NH₄) + sum(model.tracers.P) + sum(model.tracers.Z) + sum(model.tracers.DON) + sum(model.tracers.sPON) + sum(model.tracers.bPON)
total_nitrogen(::NutrientPhytoplanktonZooplanktonDetritus, model) = sum(model.tracers.N) + sum(model.tracers.P) + sum(model.tracers.Z) + sum(model.tracers.D)

function test_flat_sediment(grid, biogeochemistry; timestepper = :QuasiAdamsBashforth2)
    model = NonhydrostaticModel(; grid, biogeochemistry, 
                                  closure = nothing,
                                  timestepper)

    set_defaults!(model.biogeochemistry.sediment_model)

    set_defaults!(biogeochemistry, model)

    simulation = Simulation(model, Δt = 50, stop_time = 1day)

    intercepted_tendencies = TracerFields(keys(model.tracers), grid)

    simulation.callbacks[:intercept_tendencies] = Callback(intercept_tendencies!; callsite = TendencyCallsite(), parameters = intercepted_tendencies)

    N₀ = total_nitrogen(biogeochemistry, model) * volume(1, 1, 1, grid, Center(), Center(), Center()) + total_nitrogen(model.biogeochemistry.sediment_model) * Azᶠᶜᶜ(1, 1, 1, grid)

    run!(simulation)

    # the model is changing the tracer tendencies
    @test any([any(intercepted_tendencies[tracer] .!= model.timestepper.Gⁿ[tracer]) for tracer in keys(model.tracers)])

    # the sediment tendencies are being updated
    @test all([any(tend .!= 0.0) for tend in model.biogeochemistry.sediment_model.tendencies.Gⁿ])
    @test all([any(tend .!= 0.0) for tend in model.biogeochemistry.sediment_model.tendencies.G⁻])

    # the sediment values are being integrated
    initial_values = (N_fast = 0.0230, N_slow = 0.0807, C_fast = 0.5893, C_slow = 0.1677, N_ref = 0.0, C_ref = 0.0, N_storage = 0.0)
    @test all([any(field .!= initial_values[name]) for (name, field) in pairs(model.biogeochemistry.sediment_model.fields)])

    N₁ = total_nitrogen(biogeochemistry, model) * volume(1, 1, 1, grid, Center(), Center(), Center()) + total_nitrogen(model.biogeochemistry.sediment_model) * Azᶠᶜᶜ(1, 1, 1, grid)

    # conservations
    @test N₁ ≈ N₀

    return nothing
end

display_name(::LOBSTER) = "LOBSTER"
display_name(::NutrientPhytoplanktonZooplanktonDetritus) = "NPZD"
display_name(::SimpleMultiG) = "Multi-G"
display_name(::InstantRemineralisation) = "Instant remineralisation"

@testset "Sediment" begin
    for architecture in (CPU(), )
        grid = RectilinearGrid(architecture; size=(3, 3, 50), extent=(10, 10, 500))

        for timestepper in (:QuasiAdamsBashforth2, :RungeKutta3),
            sediment_model in (InstantRemineralisation(; grid), SimpleMultiG(; grid))
            for biogeochemistry in (NutrientPhytoplanktonZooplanktonDetritus(; grid, open_bottom = true,
                                                                               sediment_model),
                                    LOBSTER(; grid,
                                              carbonates = true, oxygen = true, variable_redfield = true, 
                                              open_bottom = true,
                                              sediment_model))

                if !(isa(sediment_model, SimpleMultiG) && isa(biogeochemistry, NutrientPhytoplanktonZooplanktonDetritus))
                    @info "Testing sediment on $(typeof(architecture)) with $timestepper and $(display_name(sediment_model)) on $(display_name(biogeochemistry))"
                    @testset "$architecture, $timestepper, $(display_name(sediment_model)), $(display_name(biogeochemistry))" test_flat_sediment(grid, biogeochemistry; timestepper)
                end
            end
        end
    end
end
