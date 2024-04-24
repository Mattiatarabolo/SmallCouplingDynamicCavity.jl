var documenterSearchIndex = {"docs":
[{"location":"functions.html","page":"Functions","title":"Functions","text":"CurrentModule = SmallCouplingDynamicCavity","category":"page"},{"location":"functions.html#The-MyAwesomePackage-Module","page":"Functions","title":"The MyAwesomePackage Module","text":"","category":"section"},{"location":"functions.html","page":"Functions","title":"Functions","text":"SmallCouplingDynamicCavity","category":"page"},{"location":"functions.html#SmallCouplingDynamicCavity.SmallCouplingDynamicCavity","page":"Functions","title":"SmallCouplingDynamicCavity.SmallCouplingDynamicCavity","text":"Module for \"SmallCouplingDynamicCavity.jl\" – A package for a Small Coupling expansion of the Dynamic Cavity method for epidemic inference.\n\nExports\n\nEpidemicModel\nROC_curve\nSI\nSIR\nSIRS\nSIS\nbethe_lattice\nrun_SCDC\nsim_epidemics\n\n\n\n\n\n","category":"module"},{"location":"functions.html#Module-Index","page":"Functions","title":"Module Index","text":"","category":"section"},{"location":"functions.html","page":"Functions","title":"Functions","text":"Modules = [SmallCouplingDynamicCavity]","category":"page"},{"location":"functions.html#Detailed-API","page":"Functions","title":"Detailed API","text":"","category":"section"},{"location":"functions.html","page":"Functions","title":"Functions","text":"Modules = [SmallCouplingDynamicCavity]\nPages   = [\"types.jl\", \"utils.jl\", \"message_passing_func.jl\", \"models/SI.jl\", \"models/SIR.jl\",\"models/SIS.jl\", \"models/SIRS.jl\"]","category":"page"},{"location":"functions.html#SmallCouplingDynamicCavity.EpidemicModel","page":"Functions","title":"SmallCouplingDynamicCavity.EpidemicModel","text":"EpidemicModel\n\nEpidemic model containing all the informations of the system.\n\nFields\n\nDisease::SmallCouplingDynamicCavity.InfectionModel: Infection model. Currently are implemented SI, SIR, SIS and SIRS infection models.\nG::Union{Vector{<:Graphs.AbstractGraph}, var\"#s31\"} where var\"#s31\"<:Graphs.AbstractGraph: Contact graph. It can be either an AbstractGraph (contact graph constant over time) or a Vector of AbstractGraph (time varying contact graph).\nN::Int64: Number of nodes of the contact graph.\nT::Int64: Number of time steps.\nν::Array{Float64, 3}: Infection couplings. It is a NVxNVx(T+1) Array where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.\nobsmat::Matrix{Int8}: Observations matrix. It is a NVx(T+1) Matrix, where obsᵗᵢ is the observation of individual i at time t: it is equal to -1.0 if not observed, 0.0 if S, 1.0 if I, 2.0 if R (only for SIR and SIRS).\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.EpidemicModel-Union{Tuple{TG}, Tuple{TI}, Tuple{TI, TG, Int64, Array{Float64, 3}, Matrix{Int8}}} where {TI<:SmallCouplingDynamicCavity.InfectionModel, TG<:(Vector{<:Graphs.AbstractGraph})}","page":"Functions","title":"SmallCouplingDynamicCavity.EpidemicModel","text":"EpidemicModel(\n    infectionmodel::TI, \n    G::TG, T::Int, \n    ν::Array{Float64, 3}, \n    obs::Matrix{Int8}) where {TI<:InfectionModel,TG<:Vector{<:AbstractGraph}}\n\nDefines the epidemic model.\n\nThis function defines an epidemic model based on the provided infection model, contact graph, time steps, infection couplings, and observation matrix.\n\n# Arguments\n- `infectionmodel`: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.\n- `G`: The contact graph. It should be a T+1 vector of AbstractGraph representing the time-varying contact graph.\n- `T`: The number of time-steps.\n- `ν`: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.\n- `obs`: The observations obsmatrix. It should be a NVx(T+1) matrix, where obsᵗᵢ is the observation of individual i at time t: it is equal to -1.0 if not observed, 0.0 if S, 1.0 if I, 2.0 if R (only for SIR and SIRS).\n\n# Returns\n- `EpidemicModel`: An [`EpidemicModel`](@ref) object representing the defined epidemic model.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.EpidemicModel-Union{Tuple{TG}, Tuple{TI}, Tuple{TI, TG, Int64, Array{Float64, 3}, Matrix{Int8}}} where {TI<:SmallCouplingDynamicCavity.InfectionModel, TG<:Graphs.AbstractGraph}","page":"Functions","title":"SmallCouplingDynamicCavity.EpidemicModel","text":"EpidemicModel(\n    infectionmodel::TI, \n    G::TG, T::Int, \n    ν::Array{Float64, 3}, \n    obs::Matrix{Int8}) where {TI<:InfectionModel,TG<:AbstractGraph}\n\nDefines the epidemic model.\n\nThis function defines an epidemic model based on the provided infection model, contact graph, time steps, infection couplings, and observation matrix.\n\nArguments\n\ninfectionmodel: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.\nG: The contact graph. It should be an AbstractGraph representing the contact graph, which is time-varying.\nT: The number of time-steps.\nν: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.\nobs: The observations matrix. It should be a NVx(T+1) matrix, where obsᵗᵢ is the observation of individual i at time t: it is equal to -1.0 if not observed, 0.0 if S, 1.0 if I, 2.0 if R (only for SIR and SIRS).\n\nReturns\n\nEpidemicModel: An EpidemicModel object representing the defined epidemic model.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.EpidemicModel-Union{Tuple{TG}, Tuple{TI}, Tuple{TI, TG, Int64, Array{Float64, 3}}} where {TI<:SmallCouplingDynamicCavity.InfectionModel, TG<:(Vector{<:Graphs.AbstractGraph})}","page":"Functions","title":"SmallCouplingDynamicCavity.EpidemicModel","text":"EpidemicModel(\n    infectionmodel::TI, \n    G::TG, T::Int, \n    ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:Vector{<:AbstractGraph}}\n\nDefine an epidemic model.\n\nThis function defines an epidemic model based on the provided infection model, time-varying contact graph, time steps, and infection couplings. It initializes the observation matrix with zeros.\n\nArguments\n\ninfectionmodel: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.\nG: The contact graph. It should be a T+1 vector of AbstractGraph representing the time-varying contact graph.\nT: The number of time-steps.\nν: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.\n\nReturns\n\nEpidemicModel: An EpidemicModel object representing the defined epidemic model.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.EpidemicModel-Union{Tuple{TG}, Tuple{TI}, Tuple{TI, TG, Int64, Array{Float64, 3}}} where {TI<:SmallCouplingDynamicCavity.InfectionModel, TG<:Graphs.AbstractGraph}","page":"Functions","title":"SmallCouplingDynamicCavity.EpidemicModel","text":"EpidemicModel(\n    infectionmodel::TI, \n    G::TG, T::Int, \n    ν::Array{Float64, 3}) where {TI<:InfectionModel,TG<:AbstractGraph}\n\nDefine an epidemic model.\n\nThis function defines an epidemic model based on the provided infection model, contact graph, time steps, and infection couplings. It initializes the observation matrix with zeros.\n\nArguments\n\ninfectionmodel: The infection model. Currently implemented models include SI, SIR, SIS, and SIRS infection models.\nG: The contact graph. It should be an AbstractGraph representing the contact graph, which is constant over time.\nT: The number of time-steps.\nν: The infection couplings. It should be a 3-dimensional array of size NVxNVx(T+1), where νᵗᵢⱼ=log(1-λᵗᵢⱼ), with λᵗᵢⱼ being the infection probability from individual i to individual j at time t.\n\nReturns\n\nEpidemicModel: An EpidemicModel object representing the defined epidemic model.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.Marginal","page":"Functions","title":"SmallCouplingDynamicCavity.Marginal","text":"Marginal\n\nMarginals pᵢ(xᵢ) and μᵢ.\n\nFields\n\ni::Int64: Index of the node i.\nm::Matrix{Float64}: (nstates)x(T+1) Matrix of marginals over time, where nstates is the number of states that the infection model has.\nμ::Vector{Float64}: T+1 Vector of marginals μᵢ over time.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.Message","page":"Functions","title":"SmallCouplingDynamicCavity.Message","text":"Message\n\nCavity messages mᵢⱼ and μᵢⱼ.\n\nFields\n\ni::Int64: Index of the node i.\nj::Int64: Index of the node j.\nm::Vector{Float64}: T+1 Vector of messages mᵢⱼ over time.\nμ::Vector{Float64}: T+1 Vector of messages μᵢⱼ over time.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.Node","page":"Functions","title":"SmallCouplingDynamicCavity.Node","text":"Node\n\nType containing all the informations of a node. \n\nFields\n\ni::Int64: Index of the node.\n∂::Vector{Int64}: List of neighbours. If the underlying contact graph is varying in time it is the union of all the neighbours over time.\n∂_idx::Dict{Int64, Int64}: Only for developers.\nmarg::SmallCouplingDynamicCavity.Marginal: Marginals of the node. It is a Marginal type.\ncavities::Vector{SmallCouplingDynamicCavity.Message}: Cavities messages entering into the node from its neigbours. It is a vector of Message, each one corresponding to a neighbour with the same order of ∂.\nρs::Vector{SmallCouplingDynamicCavity.FBm}: Only for developers.\nνs::Vector{Vector{Float64}}: Infection couplings of the neighbours against the node.\nobs::Matrix{Float64}: Observation probability matrix.\nmodel::EpidemicModel: Epidemic model. It is a EpidemicModel type.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.ROC_curve-Tuple{Vector{Float64}, Vector{Float64}}","page":"Functions","title":"SmallCouplingDynamicCavity.ROC_curve","text":"ROC_curve(marg::Vector{Float64}, x::Vector{Float64})\n\nCompute the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) based on inferred marginal probabilities and true configurations.\n\nThis function computes the ROC curve by sorting the marginal probabilities (marg) in decreasing order and sorting the true configuration (x) correspondingly. It then iterates over the sorted values to calculate the True Positive Rate (TPR) and False Positive Rate (FPR) at different thresholds, and computes the AUC.\n\nArguments\n\nmarg::Vector{Float64}: A vector containing the marginal probabilities of each node being infected at a fixed time.\nx::Vector{Float64}: The true configuration of the nodes at the same fixed time.\n\nReturns\n\nfp_rates::Vector{Float64}: Array of false positive rates corresponding to each threshold.\ntp_rates::Vector{Float64}: Array of true positive rates corresponding to each threshold.\nauc::Float64: The Area Under the Curve (AUC) of the ROC curve.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.bethe_lattice-Tuple{Int64, Int64, Bool}","page":"Functions","title":"SmallCouplingDynamicCavity.bethe_lattice","text":"bethe_lattice(z::Int, tmax::Int, startfrom1::Bool)\n\nGenerate a Bethe lattice (tree) with a specified degree and depth.\n\nArguments\n\nz::Int: The degree of the Bethe lattice.\ntmax::Int: The maximum depth of the Bethe lattice.\nstartfrom1::Bool: If true, the center of the tree is vertex 1.\n\nReturns\n\nV::Vector{Int}: A list of vertices in the Bethe lattice.\nE::Vector{Vector{Int}}: A list of edges in the Bethe lattice.\n\nDescription\n\nThis function generates a Bethe lattice (tree) with a specified degree (z) and maximum depth (tmax). The Bethe lattice is constructed by iteratively adding nodes and edges according to the specified parameters.\n\nIf startfrom1 is true, the center of the tree is vertex 1. Otherwise, the tree is constructed starting from vertex 0.\n\nThe function returns a tuple where the first element (V) is a list of vertices, and the second element (E) is a list of edges in the Bethe lattice.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.run_SCDC-Union{Tuple{TG}, Tuple{TI}, Tuple{EpidemicModel{TI, TG}, Function, Float64, Int64, Float64, Float64}} where {TI<:SmallCouplingDynamicCavity.InfectionModel, TG<:(Union{Vector{<:Graphs.AbstractGraph}, var\"#s14\"} where var\"#s14\"<:Graphs.AbstractGraph)}","page":"Functions","title":"SmallCouplingDynamicCavity.run_SCDC","text":"run_SCDC(\n    model::EpidemicModel{TI,TG},\n    obsprob::Function,\n    γ::Float64,\n    maxiter::Int64,\n    epsconv::Float64,\n    damp::Float64;\n    μ_cutoff::Float64 = -Inf,\n    callback::Function=(x...) -> nothing) where {TI<:InfectionModel,TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}\n\nRuns the Small Coupling Dynamic Cavity (SCDC) inference algorithm.\n\nThis function performs SCDC inference on the specified epidemic model, using the provided evidence (likelihood) probability function, and other parameters such as the probability of being a patient zero, maximum number of iterations, convergence threshold, damping factor, etc. It iteratively updates cavity messages until convergence or until the maximum number of iterations is reached.\n\nArguments\n\nmodel: An EpidemicModel representing the epidemic model.\nobsprob: A function representing the evidence (likelihood) probability p(O|x) of an observation O given the planted state x.\nγ: The probability of being a patient zero.\nmaxiter: The maximum number of iterations.\nepsconv: The convergence threshold of the algorithm.\ndamp: The damping factor of the algorithm.\nμ_cutoff: (Optional) Lower cut-off for the values of μ.\nn_iter_nc: (Optional) Number of iterations for non-converged messages. The messages are averaged over this number of iterations.\ndamp_nc: (Optional) Damping factor for non-converged messages.\ncallback: (Optional) A callback function to monitor the progress of the algorithm.\n\nReturns\n\nnodes: An array of Node objects representing the updated node states after inference.\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.SI","page":"Functions","title":"SmallCouplingDynamicCavity.SI","text":"struct SI <: InfectionModel\n    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities\nend\n\nThe SI struct represents the SI (Susceptible-Infected) infection model.\n\nFields\n\nεᵢᵗ: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.sim_epidemics-Union{Tuple{EpidemicModel{SI, TG}}, Tuple{TG}} where TG<:(Union{Vector{<:Graphs.AbstractGraph}, var\"#s26\"} where var\"#s26\"<:Graphs.AbstractGraph)","page":"Functions","title":"SmallCouplingDynamicCavity.sim_epidemics","text":"sim_epidemics(\n    model::EpidemicModel{SI,TG};\n    patient_zero::Union{Vector{Int},Nothing}=nothing,\n    γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}\n\nSimulates an epidemic outbreak using the SI (Susceptible-Infectious) model.\n\nArguments\n\nmodel: The SI epidemic model, encapsulating information about the infection dynamics, contact graph, and other parameters.\npatient_zero: (Optional) A vector specifying the indices of initial infected individuals. If not provided (default nothing), patient zero is selected randomly based on the probability γ.\nγ: (Optional) The probability of being a patient zero. If patient_zero is not specified and γ is provided, patient zero is chosen randomly with probability γ. If both patient_zero and γ are not provided (default nothing), patient zero is selected randomly with equal probability for each individual.\n\nReturns\n\nA matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S) and 1.0 for Infected (I).\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.SIR","page":"Functions","title":"SmallCouplingDynamicCavity.SIR","text":"struct SIR <: InfectionModel\n    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities\n    rᵢᵗ::Array{Float64, 2} # Recovery probabilities\nend\n\nThe SIR struct represents the SIR (Susceptible-Infected-Recovered) infection model.\n\nFields\n\nεᵢᵗ: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.\nrᵢᵗ: An NVxT array representing the recovery probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element rᵢᵗ[i, t] denotes the probability of node i recovering from infection at time t.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.sim_epidemics-Union{Tuple{EpidemicModel{SIR, TG}}, Tuple{TG}} where TG<:(Union{Vector{<:Graphs.AbstractGraph}, var\"#s26\"} where var\"#s26\"<:Graphs.AbstractGraph)","page":"Functions","title":"SmallCouplingDynamicCavity.sim_epidemics","text":"sim_epidemics(\n    model::EpidemicModel{SIR,TG};\n    patient_zero::Union{Vector{Int},Nothing}=nothing,\n    γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}\n\nSimulates an epidemic outbreak using the SIR (Susceptible-Infectious-Recovered) model.\n\nArguments\n\nmodel: The SIR epidemic model, encapsulating information about the infection dynamics, contact graph, and other parameters.\npatient_zero: (Optional) A vector specifying the indices of initial infected individuals. If not provided (default nothing), patient zero is selected randomly based on the probability γ.\nγ: (Optional) The probability of being a patient zero. If patient_zero is not specified and γ is provided, patient zero is chosen randomly with probability γ. If both patient_zero and γ are not provided (default nothing), patient zero is selected randomly with equal probability for each individual.\n\nReturns\n\nA matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S), 1.0 for Infected (I), and 2.0 for Recovered (R).\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.SIS","page":"Functions","title":"SmallCouplingDynamicCavity.SIS","text":"struct SIS <: InfectionModel\n    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities\n    rᵢᵗ::Array{Float64, 2} # Recovery probabilities\nend\n\nThe SIS struct represents the SIS (Susceptible-Infected-Susceptible) infection model.\n\nFields\n\nεᵢᵗ: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.\nrᵢᵗ: An NVxT array representing the recovery probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element rᵢᵗ[i, t] denotes the probability of node i recovering from infection at time t.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.sim_epidemics-Union{Tuple{EpidemicModel{SIS, TG}}, Tuple{TG}} where TG<:(Union{Vector{<:Graphs.AbstractGraph}, var\"#s26\"} where var\"#s26\"<:Graphs.AbstractGraph)","page":"Functions","title":"SmallCouplingDynamicCavity.sim_epidemics","text":"sim_epidemics(\n    model::EpidemicModel{SIS,TG};\n    patient_zero::Union{Vector{Int},Nothing}=nothing,\n    γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}\n\nSimulates an epidemic outbreak using the SIS (Susceptible-Infectious-Susceptible) model.\n\nArguments\n\nmodel: The SIS epidemic model, encapsulating information about the infection dynamics, contact graph, and other parameters.\npatient_zero: (Optional) A vector specifying the indices of initial infected individuals. If not provided (default nothing), patient zero is selected randomly based on the probability γ.\nγ: (Optional) The probability of being a patient zero. If patient_zero is not specified and γ is provided, patient zero is chosen randomly with probability γ. If both patient_zero and γ are not provided (default nothing), patient zero is selected randomly with equal probability for each individual.\n\nReturns\n\nA matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S) and 1.0 for Infected (I).\n\n\n\n\n\n","category":"method"},{"location":"functions.html#SmallCouplingDynamicCavity.SIRS","page":"Functions","title":"SmallCouplingDynamicCavity.SIRS","text":"struct SIRS <: InfectionModel\n    εᵢᵗ::Array{Float64, 2} # Autoinfection probabilities\n    rᵢᵗ::Array{Float64, 2} # Recovery probabilities\n    σᵢᵗ::Array{Float64, 2} # Loss of immunity probabilities\nend\n\nThe SIRS struct represents the SIRS (Susceptible-Infected-Recovered-Susceptible) infection model.\n\nFields\n\nεᵢᵗ: An NVxT array representing the self-infection probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element εᵢᵗ[i, t] denotes the probability of node i infecting itself at time t.\nrᵢᵗ: An NVxT array representing the recovery probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element rᵢᵗ[i, t] denotes the probability of node i recovering from infection at time t.\nσᵢᵗ: An NVxT array representing the loss of immunity probabilities over time, where NV is the number of nodes and T is the number of time-steps. Each element σᵢᵗ[i, t] denotes the probability of node i losing immunity and becoming susceptible again at time t.\n\n\n\n\n\n","category":"type"},{"location":"functions.html#SmallCouplingDynamicCavity.sim_epidemics-Union{Tuple{EpidemicModel{SIRS, TG}}, Tuple{TG}} where TG<:(Union{Vector{<:Graphs.AbstractGraph}, var\"#s26\"} where var\"#s26\"<:Graphs.AbstractGraph)","page":"Functions","title":"SmallCouplingDynamicCavity.sim_epidemics","text":"sim_epidemics(\n    model::EpidemicModel{SIRS,TG};\n    patient_zero::Union{Vector{Int},Nothing}=nothing,\n    γ::Union{Float64,Nothing}=nothing) where {TG<:Union{<:AbstractGraph,Vector{<:AbstractGraph}}}\n\nSimulates an epidemic outbreak using the SIRS (Susceptible-Infectious-Recovered-Susceptible) model.\n\nArguments\n\nmodel: The SIRS epidemic model, encapsulating information about the infection dynamics, contact graph, and other parameters.\npatient_zero: (Optional) A vector specifying the indices of initial infected individuals. If not provided (default nothing), patient zero is selected randomly based on the probability γ.\nγ: (Optional) The probability of being a patient zero. If patient_zero is not specified and γ is provided, patient zero is chosen randomly with probability γ. If both patient_zero and γ are not provided (default nothing), patient zero is selected randomly with equal probability for each individual.\n\nReturns\n\nA matrix representing the epidemic outbreak configuration over time. Each row corresponds to a node, and each column represents a time step. The values in the matrix indicate the state of each node at each time step: 0.0 for Susceptible (S), 1.0 for Infected (I), and 2.0 for Recovered (R).\n\n\n\n\n\n","category":"method"},{"location":"guide.html#Basic-use","page":"Guide","title":"Basic use","text":"","category":"section"},{"location":"guide.html","page":"Guide","title":"Guide","text":"Define an infection model through the available structures.","category":"page"},{"location":"guide.html","page":"Guide","title":"Guide","text":"# SI model with 0.0 self-infection rate, 4 individuals and 5 epidemic timesteps\njulia> infection_model = SI(0.0, 4, 5)\nSI([0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0 0.0])","category":"page"},{"location":"guide.html","page":"Guide","title":"Guide","text":"Define an epidemic model","category":"page"},{"location":"guide.html","page":"Guide","title":"Guide","text":"# SI epidemic model on a graph G with infection probability encoded by the matrix λ \njulia> model = EpidemicModel(infection_model, G, 5, log.(1 .- λ))","category":"page"},{"location":"guide.html#Sampling","page":"Guide","title":"Sampling","text":"","category":"section"},{"location":"guide.html","page":"Guide","title":"Guide","text":"# Sample the epidemic cascade specifying the patient zero as individual 1\njulia> config = sim_epidemics(model, patient_zero=[1])\n4×6 Matrix{Float64}:\n1.0  1.0  1.0  1.0  1.0  1.0\n0.0  1.0  1.0  1.0  1.0  1.0\n0.0  1.0  1.0  1.0  1.0  1.0\n0.0  0.0  0.0  1.0  1.0  1.0","category":"page"},{"location":"guide.html#Inference","page":"Guide","title":"Inference","text":"","category":"section"},{"location":"guide.html","page":"Guide","title":"Guide","text":"# Insert the observations as a matrix (-1.0 = unobserved, 0.0 = observed S, 1.0 = observed I)\njulia> model.obsmat .= [-1.0 -1.0 1.0 -1.0 -1.0 -1.0; 0.0 -1.0 -1.0 -1.0 1.0 -1.0; -1.0 -1.0 -1.0 -1.0 -1.0 -1.0;  -1.0 -1.0 -1.0 -1.0 -1.0 1.0]\n4×6 Matrix{Float64}:\n-1.0  -1.0   1.0  -1.0  -1.0  -1.0\n 0.0  -1.0  -1.0  -1.0   1.0  -1.0\n-1.0  -1.0  -1.0  -1.0  -1.0  -1.0\n-1.0  -1.0  -1.0  -1.0  -1.0   1.0\n\n# Run the inference algorithm with maximum 10 iterations, a convergence threshold of 0.1, and a damping factor of 0.0. The prior probability of being infected at time 0 is 1/4, and the observation probability obsprob is user-specified\njulia> nodes = run_SCDC(model, obsprob, 1/4, 10, 0.1, 0.0)\nConverged after 4 iterations\n4-element Vector{SmallCouplingDynamicCavity.Node{SI}}","category":"page"},{"location":"index.html#SmallCouplingDynamicCavity.jl","page":"Home","title":"SmallCouplingDynamicCavity.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Documentation for SmallCouplingDynamicCavity.jl. The GitHub repository can be found at SmallCouplingDynamicCavity.jl.git.","category":"page"},{"location":"index.html#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The purpose of SmallCouplingDynamicCavity.jl is to provide a general and computationally efficient solution for Bayesian epidemic inference and risk assessment. The package offers an efficient structure implementation for the most used epidemic models, such as Susceptible-Infected (SI), Susceptible-Infected-Recovered (SIR), Susceptible-Infected-Susceptible (SIS) and Susceptible-Infected-Recovered-Susceptible (SIRS).","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"For all these models, the package provides:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"a simulation tool, which allows to sample an epidemic outbreak with specified parameters\na statistical inference tool, which allows to obtain fully bayesian estimates of the epidemic uotbreak","category":"page"},{"location":"index.html#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"The package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"pkg> add SmallCouplingDynamicCavity","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Or, equivalently, via the Pkg API:","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"julia> import Pkg; Pkg.add(\"SmallCouplingDynamicCavity\")","category":"page"},{"location":"index.html#main-index","page":"Home","title":"Index","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Pages = [\"functions.md\"]","category":"page"}]
}
