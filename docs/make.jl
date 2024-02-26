using Documenter
using SmallCouplingDynamicCavity

push!(LOAD_PATH,"../src/")
makedocs(
    sitename = "SmallCouplingDynamicCavity.jl Documentation",
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Functions" => "functions.md"
    ],
    format = Documenter.HTML(prettyurls = false),
    modules = [SmallCouplingDynamicCavity]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "https://github.com/Mattiatarabolo/SmallCouplingDynamicCavity.jl.git",
    devbranch = "main"
)