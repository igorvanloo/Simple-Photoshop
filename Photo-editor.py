import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def change_brightness(image, value):
    img = image.copy()
    rsize, csize = len(img), len(img[0])
    for r in range(rsize):
        for c in range(csize):
            for k in range(0,3):
            #We go through each colour pixel and make sure that the original +- the value
            #Does not exceed 255 or go below 0, should it we automatically set it to 255 or 0
                if img[r,c,k] + value > 255:
                    img[r,c,k] = 255
                elif img[r,c,k] + value <= 0:
                    img[r,c,k] = 0
                else:
                    img[r,c,k] = img[r,c,k] + value
    return img 
  
def change_contrast(image, value):
    img = image.copy()
    contrast_correction_factor = (259*(value+255))/(255*(259-value))
    #This is the equation for the contrast, we will use it to simplify code later
    rsize, csize = len(img), len(img[0])
    for r in range(rsize):
        for c in range(csize):
            for k in range(0,3):
                #We use the same limits as for brightness
                if contrast_correction_factor*(img[r,c,k]-128)+128 > 255:
                    img[r,c,k] = 255
                elif contrast_correction_factor*(img[r,c,k]-128)+128 <= 0:
                    img[r,c,k] = 0
                else:
                    img[r,c,k] = contrast_correction_factor*(img[r,c,k]-128)+128              
    return img 

def grayscale(image):
    img = image.copy()
    rsize, csize = len(img), len(img[0])
    for r in range(rsize):
        for c in range(csize):
            grayscale_factor = (0.3*img[r,c,0] + 0.59*img[r,c,1] + 0.11*img[r,c,2])
            #We use the grayscale factor and apply it, no <0 or >255 precautions are needed
            #As the maximum of this factor is 255 and minimum is 0, hence a white pixel will stay white etc
            img[r,c] = grayscale_factor
    return img 

def blur_effect(image):
    new_img = image
    img = new_img.copy()
    #We create 2 img copies so that when we edit the picture we want (img), this changed value
    #Does not affect the pixels around it for their own blur calculation 
    rsize, csize = len(img), len(img[0])
    for r in range(1,rsize-1):
        for c in range(1,csize-1): 
        
            #Note that blurring requires no special precautions for greater than 255 or less than 0
            #This is because the indivdual values will never exceed 255, if all colours around are 255
            #Their sum is exactly 255 and it can never be negative
            
            img[r, c] = 0.0625 * new_img[r-1,c-1] + 0.125 * new_img[r-1,c] + 0.0625 * new_img[r-1,c+1]\
                + 0.125 * new_img[r,c-1] + 0.25 * new_img[r,c] + 0.125 * new_img[r,c+1] + 0.0625 * new_img[r+1,c-1]\
                    + 0.125 * new_img[r+1,c] + 0.0625 * new_img[r+1,c+1]
    return img

def edge_detection(image):
    new_img = image
    img = new_img.copy()
    #Same reason as for blur, we create 2 copies
    rsize, csize = len(img), len(img[0])
    for r in range(1,rsize-1):
        for c in range(1,csize-1): 
            for k in range(0,3):
                edge_test = (-1 * int(new_img[r-1,c-1,k]) -1 * int(new_img[r-1,c,k]) -1 * int(new_img[r-1,c+1,k])\
                                -1 * int(new_img[r,c-1,k]) + 8 * int(new_img[r,c,k]) -1* int(new_img[r,c+1,k]) -1 * int(new_img[r+1,c-1,k])\
                                    -1 * int(new_img[r+1,c,k]) -1 * int(new_img[r+1,c+1,k]) + 128)
                #This test function will check the total value of the pixel after the kernel for edge
                #Detection is applied, we take the int value of the RGB values to avoid the system 
                #automatically modulo 256-ing them, therefore we get a correct value, then we test 
                #if its <0 or >255, else we set the pixel as the test value
                if edge_test > 255:
                    img[r,c,k] = 255
                elif edge_test <= 0:
                    img[r,c,k] = 0
                else:
                    img[r,c,k] = edge_test     
    return img

def embossed(image):
    new_img = image
    img = new_img.copy()
    rsize, csize = len(img), len(img[0])
    for r in range(1,rsize-1):
        for c in range(1,csize-1): 
            for k in range(0,3):
                emboss_test = (-1 * int(new_img[r-1,c-1,k]) -1 * int(new_img[r-1,c,k]) + 0 * int(new_img[r-1,c+1,k])\
                                -1 * int(new_img[r,c-1,k]) + 0 * int(new_img[r,c,k]) + 1 * int(new_img[r,c+1,k]) + 0 * int(new_img[r+1,c-1,k])\
                                    + 1 * int(new_img[r+1,c,k]) + 1 * int(new_img[r+1,c+1,k]) + 128)
                #We use the same principle as edge_detection with a different kernal
                if emboss_test > 255:
                    img[r,c,k] = 255
                elif emboss_test <= 0:
                    img[r,c,k] = 0
                else:
                    img[r,c,k] = emboss_test
    return img

def rectangle_select(image, x, y):
    img = image
    mask = np.zeros((len(img),len(img[0])))
    #We reload the mask to be all zeros, incase the user would like to change their selection
    #The pixels come in as tuples therefore the top left co-ord would be (x[0],x[1]), and bottom right
    #Would be (y[0],y[1]), therefore we go from x[0] till y[0] and x[1] till y[1]
    
    for r in range(x[0], y[0]+1):
        for c in range(x[1], y[1]+1):
            #For all these values we set the mask to 1
            mask[r,c] = 1
    return mask

def magic_wand_select(image, x, thres):
    img = image.copy()
    rsize, csize = len(img), len(img[0])
    mask = np.zeros((len(img),len(img[0])))
    #Again we reset the mask to 0 if user would like to change their mask selection
    
    r, c = x[0], x[1] #For convenience sake we let the pixel be r, c
    mask[r, c] = 1 
    #We let the corresponding mask pixel be 1 as this is the starting pixel and is by default
    #going to be part of the final mask
    stack = [(r,c)] #We start the stack containing the first pixel
    
    while len(stack) != 0: #Until the stack is empty we keep checking all the pixels
        x, y = stack.pop()
        #The default value for pop() is -1 therefore we remove the last element of the list
        #This element which will be a tuple hence we let x=x co-ord and y=y co-ord
        #We then run the pixel through the 4 different directions (up, down, left, right)
        
        if x > 0: 
        #if the x co-ord of the pixel = 0 then it will skip this and will check x <rsize - 1, which
        #Checks in the opposite direction. This ensures that no pixel above x=0 will be checked
            if dist_finder(img, (r,c), (x-1,y)) <= thres and mask[x-1,y] != 1:
                #A function dist_finder takes the img, original pixel and pixel to check and returns
                #the colour distance, makes the function look more neat
                mask[x-1,y] = 1 
                stack.append((x-1,y))
                #All the below work the same: For the current popped out pixel it will go through the 
                #next 4 statements. It will check if the dist <= thres and is the mask is already 1 or not
                #If the mask is already 1 it won't be checked, this means that we will not get the infinite 
                #Runtime error of continuously checking the same pixel over and over again
                #Otherwise if the dist <= thres we will append the new pixel to the stack, if it is not true
                #Nothing happens to it it will simply be removed for the stack
                
            
        if x < rsize - 1:
        #if the x co-ord of the pixel = rsize -1 then it will skip this and x>0 
        #will already have been checked, This ensures that no pixel below x = rsize-1 will be checked
            if dist_finder(img, (r,c), (x+1,y)) <= thres and mask[x+1,y] != 1:
                mask[x+1,y] = 1 
                stack.append((x+1,y))
            
        if y > 0:
        #same logic as for x pixel the y>0 and y<csize - 1 ensures that no pixel that is outof bounds will be checked
            if dist_finder(img, (r,c), (x,y-1)) <= thres and mask[x,y-1] != 1:
                mask[x,y-1] = 1 
                stack.append((x,y-1))
        
        if y < csize - 1:
            if dist_finder(img, (r,c), (x,y+1)) <= thres and mask[x,y+1] != 1:
                mask[x,y+1] = 1 
                stack.append((x,y+1))
 
    return mask

def dist_finder(img, original_pixel, pixel_to_check):
    #We take out the values from both pixels and run the formula which will return the colour distance
    #The saves space for the magic wand functions
    
    r, c = original_pixel
    x, y = pixel_to_check
    
    red_sum = int((img[r,c,0]) + int(img[x,y,0])/2)
    red_diff = int(img[r,c,0]) - int(img[x,y,0])
    green_diff = int(img[r,c,1]) - int(img[x,y,1])
    blue_diff = int(img[r,c,2]) - int(img[x,y,2])
    
    dist = ((2 +(red_sum)/512)*((red_diff)**2) + 4 * ((green_diff)**2) + (2+((255-(red_sum))/512))*((blue_diff)**2))**(1/2)
    
    return dist

def mask_selection(new_image, image, mask):
    #This function will take the new_image after an edit has been made,
    #The original image and the mask, the mask by default is all 1 so if no mask selection is made
    #This function will do nothing, however if a mask selection is made only the pixels with corresponding
    #mask values = 1 will get the new effect
    new_img = new_image
    img = image.copy()
    mask = mask
    rsize, csize = len(img), len(img[0])
    for r in range(rsize):
        for c in range(csize):
            if mask[r,c] == 1:
                img[r,c] = new_img[r,c]
                #if the mask is one let the img = the edited image pixel, otherwise stay as unedited
    return img

def compute_edge(mask):           
    rsize, csize = len(mask), len(mask[0]) 
    edge = np.zeros((rsize,csize))
    if np.all((mask == 1)): return edge        
    for r in range(rsize):
        for c in range(csize):
            if mask[r][c]!=0:
                if r==0 or c==0 or r==len(mask)-1 or c==len(mask[0])-1:
                    edge[r][c]=1
                    continue
                
                is_edge = False                
                for var in [(-1,0),(0,-1),(0,1),(1,0)]:
                    r_temp = r+var[0]
                    c_temp = c+var[1]
                    if 0<=r_temp<rsize and 0<=c_temp<csize:
                        if mask[r_temp][c_temp] == 0:
                            is_edge = True
                            break
    
                if is_edge == True:
                    edge[r][c]=1
            
    return edge

def save_image(filename, image):
    img = image.astype(np.uint8)
    mpimg.imsave(filename,img)

def load_image(filename):
    img = mpimg.imread(filename)
    if len(img[0][0])==4: # if png file
        img = np.delete(img, 3, 2)
    if type(img[0][0][0])==np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
        img = img*255
        img = img.astype(np.uint8)
    mask = np.ones((len(img),len(img[0]))) # create a mask full of "1" of the same size of the laoded image
    img = img.astype(np.int32)
    return img, mask

def display_image(image, mask):
    # if using Spyder, please go to "Tools -> Preferences -> IPython console -> Graphics -> Graphics Backend" and select "inline"
    tmp_img = image.copy()
    edge = compute_edge(mask)
    for r in range(len(image)):
        for c in range(len(image[0])):
            if edge[r][c] == 1:
                tmp_img[r][c][0]=255
                tmp_img[r][c][1]=0
                tmp_img[r][c][2]=0
 
    plt.imshow(tmp_img)
    plt.axis('off')
    plt.show()
    print("Image size is",str(len(image)),"x",str(len(image[0])))


def menu():
    img = mask = np.array([])
    loaded_image = False #Variable which checks if an image is loaded, if yes a different menu will load
    
    while True and loaded_image == False: #Menu 1: starting menu
        User_input = input("What do you want to do ?\ne - exit\nl - load a picture\
                           \n\nYour choice: ")
        if User_input == 'e':
            break
        
        elif User_input == 'l':
            while True:
                User_picture = input("What picture would you like loaded?\nType 'e' to exit\n\nYour choice: ")
                if User_picture == 'e':
                    break
                try:
                    img, mask = load_image(User_picture)
                except FileNotFoundError:
                    print("\nError: File not found")
                else:
                    display_image(img, mask)
                    loaded_image = True
                    break
        
        else:
            print("\nPlease input a valid choice")

    while True and loaded_image == True: #Menu 2: after picture is loaded
        User_input = input("What do you want to do ?\ne - exit\nl - load a picture\
                           \ns - save the current picture \n1 - adjust brightness\
                               \n2 - adjust contrast \n3 - apply grayscale \n4 - apply blur\
                                   \n5 - edge detection \n6 - embossed \n7 - rectangle select \
                                       \n8 - magic wand select\n\nYour choice: ")
        if User_input == 'e':
            break
        
        elif User_input == 'l':
            while True:
                User_picture = input("What picture would you like loaded?\nType 'e' to exit\n\nYour choice: ")
                if User_picture == 'e':
                    break
                try:
                    img, mask = load_image(User_picture)
                except FileNotFoundError:
                    print("\nError: File not found")
                else:
                    display_image(img, mask)
                    break

        elif User_input == 's':
            while True:
                User_save = input("What would you like your image to be called?\nPlease add .jpg or .png\nType 'e' to exit\n\nYour choice: ")
                if User_save == 'e':
                    break
                try:
                    save_image(User_save, img)
                except ValueError:
                    print("\nError: You have not added an extension")
                except KeyError:
                    print("\nError: You have added an invalid extension")
                else:
                    save_image(User_save, img)
                    print("\nYou have saved your new file under the name:", User_save)
                    break
            
        elif User_input == '1':
            while True:
                User_Brightness = input("How much brighter or darker would you like your picture?\
                      \nThe range is from -255(Darkest) till +255(Brightest)\nType 'e' if you would like to exit\n\nYour choice: ")
                if User_Brightness == 'e':
                    break
                try:
                    int(User_Brightness)
                except ValueError:
                    print("Error: You have input a string instead of an integer please try again")
                else:
                    if abs(int(User_Brightness)) <= 255:
                        new_img = change_brightness(img, int(User_Brightness))
                        img = mask_selection(new_img, img, mask)
                        #Use of the maske selection function we have new_img which is the entire img
                        #brightnened by value, we then use this new_img and original image (img)
                        #and return original img with effects applied only to mask. Same thing for all functions
                        display_image(img, mask)
                        break
                    else:
                        print("\nError: Please pick a number between -255 and 255")
        
        elif User_input == '2':
            while True:
                User_contrast = input("How much contrast would you like your picture to have?\
                      \nThe range is from -255(Least) till +255(Most)\nType 'e' if you would like to exit\n\nYour choice: ")
                if User_contrast == 'e':
                    break
                try:
                    int(User_contrast)
                except ValueError:
                    print("Error: You have input a string instead of an integer please try again")
                else:
                    if abs(int(User_contrast)) <= 255:
                        new_img = change_contrast(img, int(User_contrast))
                        img = mask_selection(new_img, img, mask)
                        display_image(img, mask)
                        break
                    else:
                        print("\nError: Please pick a number between -255 and 255")
        
        elif User_input == '3':
            while True:
                User_grayscale = input("Would you like to grayscale your picture?\nPlease pick y or n\n\nYour choice: ")
                if User_grayscale == 'y':
                    new_img = grayscale(img)
                    img = mask_selection(new_img, img, mask)
                    display_image(img, mask)
                    break
                elif User_grayscale == 'n':
                    print("You have decided to not grayscale your picture")
                    break
                else:
                    print("\nError: Please pick y or n")
        
        elif User_input == '4':
            while True:
                User_blur = input("Would you like to blur your picture?\nPlease pick y or n\n\nYour choice: ")
                if User_blur == 'y':
                    new_img = blur_effect(img)
                    img = mask_selection(new_img, img, mask)
                    display_image(img, mask)
                    break
                elif User_blur == 'n':
                    print("You have decided to not blur your picture")
                    break
                else:
                    print("\nError: Please pick y or n")
                    
        elif User_input == '5':
            while True:
                User_edge_detection = input("Would you like to edge detect your picture?\nPlease pick y or n\n\nYour choice: ")
                if User_edge_detection == 'y':
                    new_img = edge_detection(img)
                    img = mask_selection(new_img, img, mask)
                    display_image(img, mask)
                    break
                elif User_edge_detection == 'n':
                    print("You have decided to not blur your picture")
                    break
                else:
                    print("\nError: Please pick y or n")
        
        elif User_input == '6':
            while True:
                User_emboss = input("Would you like to emboss your picture?\nPlease pick y or n\n\nYour choice: ")
                if User_emboss == 'y':
                    new_img = embossed(img)
                    img = mask_selection(new_img, img, mask)
                    display_image(img, mask)
                    break
                elif User_emboss == 'n':
                    print("You have decided to not emboss your picture")
                    break
                else:
                    print("\nError: Please pick y or n")
        
        elif User_input == '7':
            rsize, csize = len(img), len(img[0])
            print("This function will allow you to choose a rectangle of any size to be edited, later on please pick a top left and bottom right pixel with the following limits")
            print("Please pick a row between", 0, "and", rsize - 1)
            print("Please pick a column between", 0, "and", csize - 1)
            
            top_left_pixel_given = False #needed for second while loop to pre-emptively break out of first while loop
            bottom_right_pixel_given = False #And not activate second while loop
            
            while True:
                top_left_pixel_x = input("Please select a top left pixel row\nIf you would like to exit type 'e'\n\nYour choice: ")
                if top_left_pixel_x == 'e':
                    break
                top_left_pixel_y = input("Please select a top left pixel column\nIf you would like to exit type 'e'\n\nYour choice: ")
                if top_left_pixel_y == 'e':
                    break
                try:
                    int(top_left_pixel_x)
                    int(top_left_pixel_y)
                except ValueError:
                    print("\nError: You have not entered an integer")
                else:
                    if int(top_left_pixel_x) < 0 or int(top_left_pixel_x) >= rsize or int(top_left_pixel_y) < 0 or int(top_left_pixel_y) >= csize:
                        print("\nError: The pixel you have picked is out of range")
                        print("Please pick a row between", 0, "and", rsize - 1)
                        print("Please pick a column between", 0, "and", csize - 1)
                    else:
                        x = int(top_left_pixel_x), int(top_left_pixel_y)
                        top_left_pixel_given = True
                        print("You have chosen pixel: ",x)
                        break

            while True and top_left_pixel_given == True:        
                bottom_right_pixel_x = input("Please select a bottom right pixel row\nIf you would like to exit type 'e'\n\nYour choice: ")
                if bottom_right_pixel_x == 'e':
                    break
                bottom_right_pixel_y = input("Please select a top left pixel column\nIf you would like to exit type 'e'\n\nYour choice: ")
                if bottom_right_pixel_y == 'e':
                    break
                try:
                    int(bottom_right_pixel_x)
                    int(bottom_right_pixel_y)
                except ValueError:
                    print("\nError: You have not entered an integer")
                else:
                    if int(bottom_right_pixel_x) < 0 or int(bottom_right_pixel_x) >= rsize or int(bottom_right_pixel_y) < 0 or int(bottom_right_pixel_y) >= csize:
                        print("\nError: The pixel you have picked is out of range")
                        print("Please pick a row between", 0, "and", rsize - 1)
                        print("Please pick a column between", 0, "and", csize - 1)
                    elif int(bottom_right_pixel_x) < int(top_left_pixel_x) or int(bottom_right_pixel_y) < int(top_left_pixel_y):
                        print("\nError: Your bottom right pixel is higher or more to the left than your top left pixel")
                    else:
                        y = int(bottom_right_pixel_x), int(bottom_right_pixel_y)
                        bottom_right_pixel_given = True
                        print("You have chosen pixel: ",y)
                        break
        
            if top_left_pixel_given == True and bottom_right_pixel_given == True:
                #Only when both pixels are given value is set to true then the new mask will be given
                #Allows user to exit at any point of their pixel selection
                #Same idea for magic wand selection
                mask = rectangle_select(img, x, y)
        
        elif User_input == '8':
            rsize, csize = len(img), len(img[0])
            print("This function will allow you to choose a magical amount of pixels to be edited, later on please pick a pixel with the following limits, and a threshold")
            print("Please pick a row between", 0, "and", rsize - 1)
            print("Please pick a column between", 0, "and", csize - 1)
           
            starting_pixel_given = False
            threshold_value_given = False
            
            while True:
                pixel_x = input("Please select a top left pixel row\nIf you would like to exit type 'e'\n\nYour choice: ")
                if pixel_x == 'e':
                    break
                pixel_y = input("Please select a top left pixel column\nIf you would like to exit type 'e'\n\nYour choice: ")
                if pixel_y == 'e':
                    break
                try:
                    int(pixel_x)
                    int(pixel_y)
                except ValueError:
                    print("\nError: You have not entered an integer")
                else:
                    if int(pixel_x) < 0 or int(pixel_x) >= rsize or int(pixel_y) < 0 or int(pixel_y) >= csize:
                        print("\nError: The pixel you have picked is out of range")
                        print("Please pick a row between", 0, "and", rsize - 1)
                        print("Please pick a column between", 0, "and", csize - 1)
                    else:
                        x = int(pixel_x), int(pixel_y)
                        starting_pixel_given = True
                        break
                    
            while True and starting_pixel_given == True:        
                threshold = input("Please pick an threshold\nIf you would like to exit type 'e'\n\nYour choice: ")
                if threshold == 'e':
                        break
                try:
                    int(threshold)
                except ValueError:
                    print("\nError: You have input a string instead of a number")
                else:
                    if int(threshold) < 0:
                        print("\nYou need to pick a positive integer")
                    else:
                        thres = int(threshold)
                        threshold_value_given = True
                        break
                    
            if starting_pixel_given == True and threshold_value_given == True:      
                mask = magic_wand_select(img, x, thres)
            
        else:
            print("\nError: Please input a valid choice")
  
       
if __name__ == "__main__":
    menu()



