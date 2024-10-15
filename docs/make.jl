using GeneticDecisionTrees
using Documenter

DocMeta.setdocmeta!(GeneticDecisionTrees, :DocTestSetup, :(using GeneticDecisionTrees); recursive=true)

makedocs(;
    modules=[GeneticDecisionTrees],
    authors="Ansaar Dollie <me@ansaardollie.com>",
    sitename="GeneticDecisionTrees.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
