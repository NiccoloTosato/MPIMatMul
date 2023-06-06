set term png
set output "diffusivity1.png"
minT=0
maxT=0.9
set cbrange [minT:maxT]
plot 'diffusivity_1.dat' matrix with image

set output "diffusivity2.png"
minT=0
maxT=0.9
set cbrange [minT:maxT]
plot 'diffusivity_2.dat' matrix with image



set term png
set output "concentration.png"
minT=0
maxT=0.9
#set cbrange [minT:maxT]
plot 'concentration_1.dat' matrix with image


set term gif animate
set output "animate.gif"
frames = 34
minT=0
maxT=0.5
set cbrange [minT:maxT]
do for [i=1:frames] {
  plot 'concentration_'.i.'.dat' matrix  with image
}

