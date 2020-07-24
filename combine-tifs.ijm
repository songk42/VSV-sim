WD="/home/skim52/RSI/sim-data/sim8/";

run("Image Sequence...", "open="+WD+"rgb0/scene-001-RGB.tif sort");
saveAs("Tiff", WD+"stack-1.tif");

run("Image Sequence...", "open="+WD+"rgb2/scene-101-RGB.tif sort");
saveAs("Tiff", WD+"stack-3.tif");

run("Image Sequence...", "open="+WD+"rgb4/scene-201-RGB.tif sort");
saveAs("Tiff", WD+"stack-5.tif");

run("Image Sequence...", "open="+WD+"rgb6/scene-301-RGB.tif sort");
saveAs("Tiff", WD+"stack-7.tif");

run("Image Sequence...", "open="+WD+"rgb1/scene-051-RGB.tif sort");
saveAs("Tiff", WD+"stack-2.tif");

run("Image Sequence...", "open="+WD+"rgb3/scene-151-RGB.tif sort");
saveAs("Tiff", WD+"stack-4.tif");

run("Image Sequence...", "open="+WD+"rgb5/scene-251-RGB.tif sort");
saveAs("Tiff", WD+"stack-6.tif");

run("Image Sequence...", "open="+WD+"rgb7/scene-351-RGB.tif sort");
saveAs("Tiff", WD+"stack-8.tif");

