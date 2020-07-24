WD="/home/skim52/RSI/sim-data/sim8/";

run("Image Sequence...", "open="+WD+"rgb/scene-001-RGB.tif sort");
saveAs("Tiff", WD+"stack.tif");


