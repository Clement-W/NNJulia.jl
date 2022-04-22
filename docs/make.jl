using NNJulia
using Documenter

push!(LOAD_PATH, "../src/")

makedocs(modules=[NNJulia], sitename="NNJulia")

deploydocs(
    repo="github.com/Clement-W/Julia-DeepLearning.git",
    devbranch="gh-pages"
)



