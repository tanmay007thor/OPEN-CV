i = 1;

while i == 1
    n = camopen(0);
    sleep(200);
    im = camread(n);
    imshow(im);
    camcloseall();
end
