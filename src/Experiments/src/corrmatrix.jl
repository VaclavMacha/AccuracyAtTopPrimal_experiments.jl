@userplot CorelationMatrix

@recipe function f(
    x::CorelationMatrix;
    annotationargs = (size = 10, color = :grey),
    digits = 2
)

    y = x.args[1]
    n = size(y, 1)
    if !(isa(y, AbstractMatrix) && n == size(y, 2))
        error("Pass a square Matrix as the arg to heatmap")
    end

    grid := false # turn off the background grid
    @series begin
        seriestype := :heatmap
        y
    end

    # grid lines
    a = [0, n] .* ones(2, n + 1) .+ 0.5
    b = ones(2, n + 1) .* reshape(0:n, 1, :) .+ 0.5

    @series begin
        seriestype := :path
        primary := false
        linecolor --> :lightgrey
        hcat(a, b), hcat(b, a)
    end

    # values
    @series begin
        seriestype := :scatter
        seriesalpha := 0
        primary := false
        series_annotations := makeannotation.(y; digits, annotationargs)[:]
        repeat(1:n, inner = n), repeat(1:n, outer = n)
    end
end

function makeannotation(val; annotationargs = (), digits = 2)
    if ismissing(val)
        val = ""
    else
        val = round(val; digits)
        val = isinteger(val) ? Int(val) : val
    end
    return text(val, annotationargs...)
end
