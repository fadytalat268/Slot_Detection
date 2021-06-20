from vacancy_inference import check_vacancy,vacancy_model_init,draw_classification
from handiacapped import compare_hist_list
from dip_project import init_marking_points_model, image_predict_marking_points
from slot_functions import inference_slots




point_model = None
vacancy_m = None




def pipeline_init():
    global point_model, vacancy_m
    
    point_model = init_marking_points_model();
    
    vacancy_m = vacancy_model_init()
    






def Slot_Annotation_pipeline(in_image):
    '''
    1. get the marking points from prediction mode
    2. pair and form slots if found
    3. check slots vacancy -if found- using vacancy model
    4. if vacant slot found check if handi-capped 
    '''
    # get predicted points
    pred_points, out_image = image_predict_marking_points(in_image, point_model)
    
    # detect slots if found
    out0, slots_dict = inference_slots(pred_points[0])
    
    # Check vacancy of slots
    vacancy_states = check_vacancy(in_image, slots_dict, vacancy_m)
    for state in vacancy_states:
        if (state[1] == 'vacant' or state[1] == 'handicapped' ):
            handi = compare_hist_list(state[0])
            if handi:
                state[1] = 'handicapped'
    
        # draw each slot type on it
        draw_classification(out_image,state[2],state[1])
    

    
    return out_image
    
    
    
    
    
    
    
    
    
