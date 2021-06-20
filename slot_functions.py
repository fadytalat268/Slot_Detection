# -*- coding: utf-8 -*-
"""Slot_Functions

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GMjCg4m4yMY_wO_JHY9zSDIW2wghUDYX

## Calculate Direction(angle)
"""

def calc_angle(p1,p2):
  
  y_diff = p2[1]- p1[1]
  x_diff = p2[0] - p1[0]
  #print(x_diff,y_diff)
  if x_diff == 0 and y_diff ==0 :
    return 0
  theta = math.atan(abs( y_diff/x_diff)) * (180/math.pi)
 # print(theta)
  
  if x_diff >=0 and y_diff >=0:
    return theta
  elif x_diff < 0 and y_diff >=0:
    return 180- theta
  elif x_diff < 0 and y_diff < 0:
    return - 180 + theta
  elif x_diff > 0 and y_diff <0:
    return -1* theta
  else:
    return theta

def calc_angle_vector(p1):
  
  y_diff = p1[1]
  x_diff = p1[0] 
  if x_diff == 0 and y_diff ==0 :
    return 0
  #print('x w y',x_diff,y_diff)
  theta = math.atan(abs( y_diff/x_diff)) * (180/math.pi)
  
  if x_diff >=0 and y_diff >=0:
    return theta
  elif x_diff < 0 and y_diff >=0:
    return 180- theta
  elif x_diff < 0 and y_diff < 0:
    return  180+theta        ###################270 - theta
  elif x_diff > 0 and y_diff <0:
    return -1* theta
  else:
    return theta

"""## Calculate difference in direction"""

def abs_direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs(abs(direction_a) - abs(direction_b))
    #print('elfr2',diff)
    #return diff if diff < 180 else -1* diff
    return diff

def direction_diff(direction_a, direction_b):
    """Calculate the angle between two direction."""
    diff = abs((direction_a) - (direction_b))
    #print('elfr2',diff)
    #return diff if diff < 180 else -1* diff
    return diff

"""## Determine Point Shape"""

none = 0
l_down = 2
t_down = 3
t_middle = 4
t_up = 5
l_up = 6 
BRIDGE_ANGLE_DIFF = 1.52 # 0.09757113548987695 + 0.1384059287593468
SEPARATOR_ANGLE_DIFF = 0.284967562063968 + 0.1384059287593468

def detemine_point_shape(point, vector):
    """Determine which category the point is in."""
    #print('a,b',point[1],point[2],'vector01',vector[0],vector[1])
    vec_direct =calc_angle_vector(vector) #*(180/math.pi)
   # vec_direct_up = calc_angle_vector([-1*vector[0], vector[1]]) #*(180/math.pi)
    vec_direct_up = 180 - vec_direct
    vec_direct_down = calc_angle_vector([vector[0], -1*vector[1]]) #*(180/math.pi)
    marking_direction =  calc_angle ([point[1],point[2]],[point[3],point[4]]) #*(180/math.pi)
    print('vector',vec_direct,'mark',marking_direction,'v up',vec_direct_up,'v down',vec_direct_down)
    #point[5] = 0
    if point[5] < 50:
        #print('midd',abs_direction_diff(vec_direct, marking_direction),vec_direct,marking_direction)
        if abs_direction_diff(vec_direct, marking_direction) >60 and  abs_direction_diff(vec_direct, marking_direction) <135:  # < angle 13.5
            print('ana middle t')
            return t_middle
        #print( 'up',direction_diff(vec_direct_up, marking_direction))
        #if direction_diff(vec_direct_up, marking_direction) < 20: # angle 24.3
          #  print('ana up t')
           # return t_up
        #print('down',direction_diff(vec_direct_down, marking_direction))
        if direction_diff(vec_direct_down, marking_direction) <50:
            print('ana down t')
            return t_down
    
    else:
      #print('point',point,'vector',vec_direct,'mark',marking_direction,'v up',vec_direct_up,'v down',vec_direct_down)
      difference = marking_direction - vec_direct_down 
      #print('diff',difference)
      if difference> 170:
          print('ana down l')
          return l_down
      if difference < 170:
          print('ana up l')
          return l_up
    return none

"""## Pair Marking Points to form slots (can these two make a slot)"""

def pair_marking_points(point_a, point_b):
    """See whether two marking points form a slot."""
    vector_ab = np.array([point_b[1] - point_a[1], point_b[2] - point_a[2]])
    vector_ab = vector_ab / np.linalg.norm(vector_ab)
    point_shape_a = detemine_point_shape(point_a, vector_ab)
    point_shape_b = detemine_point_shape(point_b, -vector_ab)
    none = 0
    l_down = 2
    t_down = 3
    t_middle = 4
    t_up = 5
    l_up = 6 
    #print("point shapes",point_shape_a,point_shape_b)
    if ( point_shape_a == l_up and point_shape_b == l_up)  :
      return [1,point_shape_a,point_shape_b]
    if ( point_shape_a == l_down and point_shape_b == l_down)  :
      return [1,point_shape_a,point_shape_b]
    if  ( point_shape_a == t_middle and point_shape_b == t_middle ) :
      return [1,point_shape_a,point_shape_b]
    if (point_shape_a == t_down and point_shape_b == t_down) :  
      vect1 = np.array([point_a[3] - point_a[1], point_a[4] - point_a[2]])
      vect1 = vect1 / np.linalg.norm(vect1)
      vect2 = np.array([point_b[3] - point_b[1], point_b[4] - point_b[2]])
      vect2 = vect2 / np.linalg.norm(vect2)
      dot_product = (np.dot(vect1, vect2))
      print('dot',dot_product)
      if dot_product > 0.3:
        return [1,point_shape_a,point_shape_b]
      else:
        return [0]
    if (point_shape_a == l_up and point_shape_b == t_middle) or (point_shape_b == l_up and point_shape_a == t_middle):
      return [1,point_shape_a,point_shape_b]
    if (point_shape_a ==t_middle  and point_shape_b ==l_down)  or (point_shape_b ==t_middle  and point_shape_a ==l_down):
      return [1,point_shape_a,point_shape_b]
    if (point_shape_a ==t_middle  and point_shape_b ==l_up)  or (point_shape_b ==t_middle  and point_shape_a ==l_up):
      return [1,point_shape_a,point_shape_b]
    if point_shape_a == t_up and point_shape_b == l_up or (point_shape_b == t_up and point_shape_a == l_up ):
      return [1,point_shape_a,point_shape_b]
    #if point_shape_a == t_up and point_shape_b == t_up:
     # return 1
    if (point_shape_a == l_down and point_shape_b == l_down) :
      return [1,point_shape_a,point_shape_b]
    if (point_shape_a == l_down and point_shape_b == t_down) or (point_shape_b == l_down and point_shape_a == t_down):
      return [1,point_shape_a,point_shape_b]
    if (point_shape_a == t_down and point_shape_b == l_up) or (point_shape_b == t_down and point_shape_a == l_up):
      return [1,point_shape_a,point_shape_b]
    
    return [0]


perpend_min = 120
perpend_max = 195
parallel_min = 340
parallel_max = 370


def inference_slots(marking_points):
    num_detected = marking_points.shape[0]
    print('numebr of points', num_detected)
    perpen_parallel = 0  # perpendicular = 1 , parallel = 2

    slot = {}
    slots = []

    marks_set = set()
    marks_list = []
    slots_list = []
    for i in range(num_detected - 1):
        for j in range(i + 1, num_detected):
            # print (i,j)
            point_i = marking_points[i]
            point_j = marking_points[j]
            # print('poin i and j',point_i,point_j)
            # Step 1: distance check
            distance = calc_point_squre_dist(point_i, point_j)
            print('distance', distance)
            # print ("distance",distance)
            if not (perpend_min <= distance <= perpend_max
                    or parallel_min <= distance <= parallel_max):
                continue

            # Step 2: pass through
            if pass_through_third_point(marking_points, i, j) and pass_through_third_point(marking_points, j, i):
                continue
            result = pair_marking_points(point_i, point_j)
            # print("result",result)
            if (perpend_min <= distance <= perpend_max):
                perpen_parallel = 1  # perpendiculer
            elif (parallel_min <= distance <= parallel_max):
                perpen_parallel = 2  # parallel

            if len(result) <= 1:
                if result == 0:
                    print('no pair')
                    continue
            else:
                if result[0] == 1:
                    print('paaaaaaaaaaaaair', i, j, marking_points[i][1] - marking_points[j][1],
                          marking_points[i][2] - marking_points[j][2])
                    print('........distnace', distance)
                    first_point, second_point = order_pair(marking_points[i], marking_points[j])
                    print('done order')
                    # if abs(marking_points[i][1] - marking_points[j][1]) <150 and abs(marking_points[i][2] - marking_points[j][2])<150:
                    test1 = {'x1': first_point[1], 'y1': first_point[2], 'dir_x1': first_point[3],
                             'dir_y1': first_point[4],
                             'x2': second_point[1], 'y2': second_point[2], 'dir_x2': second_point[3],
                             'dir_y2': second_point[4],
                             'type': perpen_parallel, 'type1': result[1], 'type2': result[2]}

                    marks_list.append(marking_points[i])
                    marks_list.append(marking_points[j])
                    mark_set = set(marks_list)
                    mark_list = list(mark_set)
                    slots_list.append((i + 1, j + 1, perpen_parallel))
                    slot.update(test1.copy())
                    slots.append(slot.copy())

    slots2 = {'marks': marks_list, 'slots': slots_list}
    return slots, slots2


def order_pair(point1,point2):
  if abs(point2[2]-point1[2])<20:
    if point2[1] > point1[1]:
      return point2,point1
    else:
      return point1,point2
  else:
    if max(point2[1] , point1[1]) >255:
      if point2[2] < point1[2]:
        return point2,point1
      else:
        return point1,point2
    elif max(point2[1] , point1[1]) <255:
      print('first half')
      if point1[2]> point2[2] :
        print('1,2')
        return point1,point2
      else:
        return point2,point1

def visualize_slot(image,prediction):
    
    slots = prediction
    print('aana hrsm',slots)
    plt.imshow(image.permute(1, 2, 0))
    for i in range(len(slots)):
     #   plt.plot([pre[i][1]],[pre[i][2]],'o')
      #  plt.plot([pre[i][3]],[pre[i][4]],'o')
       x = [slots[i]['x1'],slots[i]['x2']]
       y= [slots[i]['y1'],slots[i]['y2']]
       mark1_x = [slots[i]['x1'],slots[i]['dir_x1']]
       mark1_y = [slots[i]['y1'],slots[i]['dir_y1']]
       mark2_x = [slots[i]['x2'],slots[i]['dir_x2']]
       mark2_y = [slots[i]['y2'],slots[i]['dir_y2']]
      
       p1 = '1'
       p2 = '2'
       plt.text(slots[i]['x1']+15, slots[i]['y1']  , p1,fontsize=10 , color='white')
       plt.text(slots[i]['x2']+15, slots[i]['y2'] , p2,fontsize=10 , color='white')
       txt = 'slot '+ str(i+1)
       plt.text(max(slots[i]['x1'],slots[i]['x2']),(slots[i]['y1']+slots[i]['y2']) /2 +15 , txt,fontsize=12 , color='red')
       #dir_x = [slots[i]['dir_x1'],slots[i]['dir_x2']]
       #dir_y = [slots[i]['dir_y1'],slots[i]['dir_y2']]
       plt.plot(x,y,mark1_x,mark1_y,mark2_x, mark2_y,linewidth=3, markersize=3, color='green')
       
    #plt.show()

"""##### example 

"""

# k = 3906
# model= model.eval()
# start = timeit.default_timer()
# predict_awal2 = predict(park_dataset[k]['image'].reshape((1,3,512,512)).to(device))
# stop = timeit.default_timer()
# print('Time: ', stop - start)  
# predict_ba3d2= get_predicted_points(predict_awal2,50)
# predict_ba3d2 = non_maximum_suppression(predict_ba3d2)

# res3,res2_temp = inference_slots(predict_ba3d2[0])
# visualize_after_thres2(park_dataset[k]['image'],predict_ba3d2[0])
# visualize_slot(park_dataset[k]['image'],res3)


