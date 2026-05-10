# collect files -- Code entirely by Prof. Kayar
# import Pkg; Pkg.add("GLMakie")
import Pkg; Pkg.add("CSV")
using GLMakie, CSV, DataFrames

function frame_number(file)
 m = match(r"sph_(\d+)\.csv$", basename(file))
 return parse(Int, m.captures[1])
end

files = sort(filter(f-> endswith(f, ".csv"), readdir("output";
join=true));
 by = frame_number)

isempty(files) && error("No CSV files found.")

# read first frame
df = CSV.read(files[1], DataFrame; header=false)
rename!(df, [:x, :y])

xs = Observable(df.x)
ys = Observable(df.y)

fig = Figure()
ax = Axis(fig[1,1], limits=(0,1,0,1))
scatter!(ax, xs, ys; markersize=20, color=:blue, alpha=0.6)

display(fig)

# animation loop
while true
for file in files
 df = CSV.read(file, DataFrame; header=false)
 rename!(df, [:x, :y])

 if nrow(df) == 0
 continue
 end

 xs[] = df.x
 ys[] = df.y

 sleep(0.001)
 yield()
end
end 