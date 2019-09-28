# tf.transformations alternative is not yet available in tf2
from tf.transformations import *
import numpy as np
import math
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import Image

#segment_angle_by_kmeans segemnts angles into 2 groups, horizontal and vertical (0/180 degrees for vertical lines and 90 degrees for horizonal lines in Hesse normal form). This is so that intersections are only found between perpendicular lines.
def segment_by_angle_kmeans(lines, k=2, **kwargs):
    # Define kmeans criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER #stop if either minimum accuracy and maximum iterations are met
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS) #This flag is used to specify how initial centers are taken.
    attempts = kwargs.get('attempts', 10) #Flag to specify the number of times the algorithm is executed using different initial labellings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.
    angles = np.array([line[0][1] for line in lines]) #create array of line angles

    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:] #lables is index of centers, in this case 2
    labels = labels.reshape(-1)  # transpose to row vector

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line) #add lines to 2 groups, horizontal and vertical
    segmented = list(segmented.values()) #create list of segmented lines
    return segmented

#find the point where any 2 lines meet
def intersection(line1, line2):
    rho1, theta1 = line1[0] #define line in Hesse normal form
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b) #use linear algebra to solve for intersections
    x0, y0 = int(np.round(x0)), int(np.round(y0)) #round intersection to nearest integer(pixel)
    return [[x0, y0]]

#repeats intersection function for all lines
def segmented_intersections(lines):
    intersections = []
    for i, group in enumerate(lines[:-1]): #iterate through the lines
        for next_group in lines[i+1:]: #move to next group
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2)) #append each intersection for each pair of lines

    return intersections

def brick_boi(image):
    try:
        x_offset = -0.057 #x and y offset define the distance between the center of the gripper and the center of the iage returned by the camera
        y_offset = -0.015
        Angular_offset = 0 #angular offset defines the angular difference between the gripper and the camera
        img = image[:, 216:481] #crop image to remove grippers from view
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #convert colourspace to RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #desaturate image
        blur = cv2.medianBlur(gray, 5) #blur image
        adapt_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C #threshold value is the weighted sum of neighborhood values where weights are a Gaussian window.
        thresh_type = cv2.THRESH_BINARY_INV #define threshold type, hard border between pixels
        bin_img = cv2.adaptiveThreshold(blur, 255, adapt_type, thresh_type, 11, 2) #11 is the size of the pixel neighbourhood used to find the threshold value. 2 is the constant subtracted from the mean
        rho, theta, thresh = 1, np.pi/180, 50 #define angular accuracies and threshold for hough transform
        lines = cv2.HoughLines(bin_img, rho, theta, thresh) #use houghlines alogrithm to identify lines
        segmented = segment_by_angle_kmeans(lines) #return lines segemented by horizontal and vertical
        intersections = segmented_intersections(segmented) #return the segmented intersections
        for i in range (0,len(intersections)): #create an array of intersections and plot them on the image
            array = np.array(intersections[i])
            plt.plot(array[0,0], array[0,1], 'g+')  #plot with green +
        newintersections = [] #create an empty array
        for i in range (0,len(intersections)):
            newintersections.append(intersections[i][0]) #format array to allow kmeans to run
        kmeans = KMeans(n_clusters=4, random_state=0).fit(newintersections) #find and return 4 averaged clusters of coordinates
        centers4=kmeans.cluster_centers_
        centers=centers4[0:3] #discard final coordinate
        line1 = [centers[0,0]-centers[1,0], centers[0,1]-centers[1,1]] #define 3 lines joining each average centers
        line2 = [centers[0,0]-centers[2,0], centers[0,1]-centers[2,1]]
        line3 = [centers[1,0]-centers[2,0], centers[1,1]-centers[2,1]]
        line1len = math.sqrt((line1[0]**2)+(line1[1]**2)) #calculate length of the 3 lines using pythagoras
        line2len = math.sqrt((line2[0]**2)+(line2[1]**2))
        line3len = math.sqrt((line3[0]**2)+(line3[1]**2))
        linelen = [line1len, line2len, line3len] #create list of line lengths
#we dont know which line will be the second longest, we need to identify it as it is of known length and will have higher accuracy than the shortest line.
        values = [0,1,2] #create variables to categorise lines
        values.remove(linelen.index(max(linelen))) #remove the largest line from values
        values.remove(linelen.index(min(linelen))) #remove the smallest line from values
        if (values[0] == 0): #check value and define 2nd longest line
            shortestline = [centers[0,0]-centers[1,0], centers[0,1]-centers[1,1]]
            angle = np.arctan2(shortestline[0], shortestline[1]) #calculate angle of line
            scalefactor = 0.2/line1len #define scale factor using known length of 0.2 meters

        if (values[0] == 1): #repeat for if line is found at index 1
            shortestline = [centers[0,0]-centers[2,0], centers[0,1]-centers[2,1]]
            angle = np.arctan2(shortestline[0], shortestline[1])
            scalefactor = 0.2/line2len

        if (values[0] == 2): #repeat for if line is found at index 2
            shortestline = [centers[1,0]-centers[2,0], centers[1,1]-centers[2,1]]
            angle = np.arctan2(shortestline[0], shortestline[1])
            scalefactor = 0.2/line3len

        if (angle >= math.pi/2): #subtract 180 degrees from angle if angle is greater than 90 degrees
            angle = angle - math.pi

        if (angle <= -math.pi/2): #add 180 degrees from angle if angle is less than 90 degrees
            angle = angle + math.pi
        averagecenters = [0,0]
        angle = -angle + Angular_offset #add angular offset to original angle

        averagecenters[0] = (centers4[0,0] + centers4[1,0]+ centers4[2,0]+ centers4[3,0])/4 #find center of brick (x coordinate)
        averagecenters[1] = (centers4[0,1] + centers4[1,1]+ centers4[2,1]+ centers4[3,1])/4 #find center of brick (y coordinate)

        yerror = y_offset -(scalefactor * (averagecenters[0]- img.shape[1]/2)) #calculate x offset by subtracting the x coordinate of the middle of the image from the x coordinate of the center of the brick then multiplying it by the scale factor
        xerror = x_offset - scalefactor * (averagecenters[1]- img.shape[0]/2)  #calculate y offset by subtracting the y coordinate of the middle of the image from the y coordinate of the center of the brick then multiplying it by the scale factor
        print("+++++++++++++",scalefactor * (averagecenters[1]- img.shape[0]/2)) #output to terminal
        plt.plot(averagecenters[0],averagecenters[1], 'ro') #plot center of brick
        plt.plot(img.shape[1]/2,img.shape[0]/2, 'r+') #plot center of image

        plt.imshow(img) #plot image of brick
        #plt.show()
        plt.savefig('/home/petar/catkin_ws/src/WELL/scripts/final/perception_test/plot.png') #save final plot
        plt.clf() #clear previous plot
        im1=Image.open('/home/petar/catkin_ws/src/WELL/scripts/final/perception_test/plot.png') #open saved plot
        im1=im1.resize((800,600),Image.BICUBIC) #use bicubic interpolation to resize image to fit screen
        im1.save('/home/petar/catkin_ws/src/WELL/scripts/final/perception_test/plot.png') #save resized image
        for i in range (0, len(centers)):
            plt.plot(centers[i][0],centers[i][1], 'g+') #plot all centers
        return(xerror, yerror, angle) #output x error, y error, and angular offset

    except: #if no brick is found, return a list of zeros
        return([0,0,0])

def calculate_brick_locations():
	overhead_orientation = np.array([-0.0249590815779,0.999649402929,0.00737916180073,0.00486450832011])
	#overhead_orientation = np.array([0,0,0,0])
	brick_dimensions = [0.2, 0.04, 0.09]  #Lenght, Width, Height. Should width be 0.04 or 0.062??? It was 0.062, Hugo changed it.

	robot_offset = [0.6, 0.4, -0.22, 0]
	bricks_per_layer = 5
	num_layers = 3
	gap = 0.006 # Gap between each brick corner and corner of polygon.

	adjacent_length = 0.1*((brick_dimensions[0]/2+gap)/(math.tan(math.radians(360/(2*bricks_per_layer)))))

	well_centre_to_brick_origin_lenght = adjacent_length + brick_dimensions[1]/2

	angles = np.zeros((bricks_per_layer, 1))

	for i in range(bricks_per_layer):
		theta = 360/bricks_per_layer
		angles[i] = robot_offset[3] + i*theta

	num_bricks = bricks_per_layer * num_layers
	brick_locations = np.zeros(shape=(num_bricks, 7))

	for i in range(num_layers):
		for j in range(bricks_per_layer):
				brick_number = i*bricks_per_layer+j
				brick_locations[brick_number, 0] = 0.1*(math.degrees(math.sin(math.radians(angles[j] + (i%2)*(360/(2*bricks_per_layer)) + (robot_offset[3])))) * well_centre_to_brick_origin_lenght) + robot_offset[0]  # Fudge factor of 0.1, we dont know why exactly this came in (perhaps degree/radian conversion) but it is accurate.
				brick_locations[brick_number, 1] = 0.1*(math.degrees(math.cos(math.radians(angles[j] + (i%2)*(360/(2*bricks_per_layer)) + (robot_offset[3])))) * well_centre_to_brick_origin_lenght) + robot_offset[1]
				brick_locations[brick_number, 2] = (i+0.5)*brick_dimensions[2] + robot_offset[2]
				brick_locations[brick_number, 3:7]= quaternion_multiply(quaternion_from_euler(0,0,-math.radians(angles[j] + (i%2)*(360/(2*bricks_per_layer)))), overhead_orientation)  #quaternion_from_euler(0,0,-math.radians(angles[j] + i*(360/(2*bricks_per_layer))))
#quaternion_multiply(quaternion_from_euler(0,0,math.radians(angles[j] + i*(360/(2*bricks_per_layer)))),overhead_orientation)

	brick_locations_optimised = np.zeros(shape=(num_bricks, 7))

	route = list(range(bricks_per_layer))

	optimised_route = [0]*bricks_per_layer
	flag = 0

	for i in range(bricks_per_layer):
		if flag == 0:
			optimised_route[i] = min(route)
			route.remove(min(route))
			flag = 1
		elif flag == 1:
			optimised_route[i] = max(route)
			route.remove(max(route))
			flag = 0

	for i in range(num_layers):
		for j in range(bricks_per_layer):
			old_brick_number = i*bricks_per_layer+j
			new_brick_number = i*bricks_per_layer+optimised_route[j]
			brick_locations_optimised[old_brick_number] = brick_locations[new_brick_number]
	return brick_locations_optimised


def calculate_brick_locations_dual():
	overhead_orientation = np.array([-0.0249590815779,0.999649402929,0.00737916180073,0.00486450832011])
	overhead_orientation_r = np.array([0.0249590815779,0.999649402929,0.00737916180073,-0.00486450832011])
	#overhead_orientation = np.array([0,0,0,0])
	brick_dimensions = [0.2, 0.062, 0.09]  #Lenght, Width, Height. Should width be 0.04 or 0.062??? It was 0.062, Hugo changed it.

	robot_offset = [0.59, 0, 0.13, 0] #0.15-0.24
	bricks_per_layer = 4
	num_layers = 3
	gap = -0.1 # Gap between each brick corner and corner of polygon.

	adjacent_length = 0.1*((brick_dimensions[0]/2+gap)/(math.tan(math.radians(360/(2*bricks_per_layer)))))

	well_centre_to_brick_origin_lenght = adjacent_length + brick_dimensions[1]/2

	angles = np.zeros((bricks_per_layer, 1))

	if (bricks_per_layer % 2) == 1:
		print('bricks per layer must be even for dual arms. Please change.')

	for i in range(bricks_per_layer):
		theta = 360/bricks_per_layer
		angles[i] = robot_offset[3] + i*theta-45

	num_bricks = int(bricks_per_layer/2 * num_layers)
	brick_locations_left = np.zeros(shape=(num_bricks, 7))
	brick_locations_right = np.zeros(shape=(num_bricks, 7))

	for i in range(num_layers):
		for j in range(int(bricks_per_layer/2)):
			brick_number = i*int(bricks_per_layer/2)+j
			brick_locations_right[brick_number, 0] = 0.1*(math.degrees(math.sin(math.radians(angles[j] + (i % 2)*(360/(2*bricks_per_layer)) + (robot_offset[3])))) * well_centre_to_brick_origin_lenght) + robot_offset[0]  # Fudge factor of 0.1, we dont know why exactly this came in (perhaps degree/radian conversion) but it is accurate.
			brick_locations_right[brick_number, 1] = 0.1*(math.degrees(math.cos(math.radians(angles[j] + (i % 2)*(360/(2*bricks_per_layer)) + (robot_offset[3])))) * well_centre_to_brick_origin_lenght) + robot_offset[1]
			brick_locations_right[brick_number, 2] = (i+0.5)*brick_dimensions[2] + robot_offset[2]
			brick_locations_right[brick_number, 3:7] = quaternion_multiply(quaternion_from_euler(0, 0, -math.radians(angles[j] + (i%2)*(360/(2*bricks_per_layer)))), overhead_orientation_r) #quaternion_multiply(quaternion_from_euler(0, 0, -math.radians(angles[j] + (i % 2) * (360 / (2 * bricks_per_layer)))), overhead_orientation)  # quaternion_from_euler(0,0,-math.radians(angles[j] + i*(360/(2*bricks_per_layer))))
			# quaternion_multiply(quaternion_from_euler(0,0,math.radians(angles[j] + i*(360/(2*bricks_per_layer)))),overhead_orientation)
		for j in range(int(bricks_per_layer/2)):
			brick_number = i*int(bricks_per_layer/2) + j
			brick_locations_left[brick_number, 0] = 0.1 * (math.degrees(math.sin(math.radians(angles[j + int(bricks_per_layer/2)] + (i % 2) * (360 / (2 * bricks_per_layer)) + (robot_offset[3])))) * well_centre_to_brick_origin_lenght) + robot_offset[0]  # Fudge factor of 0.1, we dont know why exactly this came in (perhaps degree/radian conversion) but it is accurate.
			brick_locations_left[brick_number, 1] = 0.1 * (math.degrees(math.cos(math.radians(angles[j + int(bricks_per_layer/2)] + (i % 2) * (360 / (2 * bricks_per_layer)) + (robot_offset[3])))) * well_centre_to_brick_origin_lenght) + robot_offset[1]
			brick_locations_left[brick_number, 2] = (i + 0.5) * brick_dimensions[2] + robot_offset[2]
			brick_locations_left[brick_number, 3:7] = quaternion_multiply(quaternion_from_euler(0, 0, -math.radians(angles[j] + (i%2)*(360/(2*bricks_per_layer)))), overhead_orientation) #quaternion_multiply(quaternion_from_euler(0, 0, -math.radians(angles[j + int(bricks_per_layer/2)] + (i % 2) * (360 / (2 * bricks_per_layer)))), overhead_orientation)  # quaternion_from_euler(0,0,-math.radians(angles[j] + i*(360/(2*bricks_per_layer))))
			# quaternion_multiply(quaternion_from_euler(0,0,math.radians(angles[j] + i*(360/(2*bricks_per_layer)))),overhead_orientation)
	brick_locations_right_optimised = np.zeros(shape=(num_bricks, 7))
	brick_locations_left_optimised = np.zeros(shape=(num_bricks, 7))

	route = list(range(int(bricks_per_layer/2)))

	optimised_route = [0]*int(bricks_per_layer/2)
	flag = 0

	for i in range(int(bricks_per_layer/2)):
		if flag == 0:
			optimised_route[i] = min(route)
			route.remove(min(route))
			flag = 1
		elif flag == 1:
			optimised_route[i] = max(route)
			route.remove(max(route))
			flag = 0

	for i in range(num_layers):
		for j in range(int(bricks_per_layer/2)):
			old_brick_number = i*int(bricks_per_layer/2)+j
			new_brick_number = i*int(bricks_per_layer/2)+optimised_route[j]
			brick_locations_right_optimised[old_brick_number] = brick_locations_right[new_brick_number]
			brick_locations_left_optimised[old_brick_number] = brick_locations_left[new_brick_number]

	return [brick_locations_right_optimised, brick_locations_left_optimised]

def plan_path(start, end):

	number_of_steps_curve = 2.00
	number_of_steps_drop = 2.00
	lowering_height = 0.2

	path_sum = number_of_steps_curve+number_of_steps_drop
	path = np.zeros((int(path_sum), 3)) # initialise array of zeros
	step_xyz = [0,0,0] #initialise step

	for i in range (0,3):
		path[0, i] = start[i] #set first step as start position
		step_xyz[i] = (end[i]-start[i])/number_of_steps_curve #define step distance

	step_xyz[2] = (((start[2]+ (end[2]+lowering_height)))/float(number_of_steps_curve)) #define z step distance

	for i in range(1,(int(number_of_steps_curve) + int(number_of_steps_drop))): #incrememnt x and y path according to step_xyz

		if (i <= number_of_steps_curve):
			path[i,0] = float(path[i-1,0]+step_xyz[0])
			path[i,1] = path[i-1,1]+step_xyz[1]
			path[i,2] = ((i**(1./3)) / ((number_of_steps_curve-1)**(-(2./3)))) * (step_xyz[2]) #incremment z path according to curve equation

		else:                                                           #keep x and y the same
			path[i,0] = float(path[i-1,0])
			path[i,1] = path[i-1,1]
			path[i,2] = path[i-1,2] - (lowering_height / number_of_steps_drop)      #lower z according to drop height

	path[int(number_of_steps_curve) + int(number_of_steps_drop)-1] = end                      #ensure final value is equal to target (without this for low values of number_steps_drop the final path can have a small z error)

	return path
