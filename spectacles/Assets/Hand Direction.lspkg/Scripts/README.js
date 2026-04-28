/*
README: Hand Direction

Note: 
- This asset works best on the back / world-facing camera
- Hand tracking works best with good lighting

To use Hand Direction: 

1.  Add the Hand_Direction prefab to the Objects panel under the main camera.

2.  In the Inspector panel for the Hand Direction Detector scene object, select
    a palm direction from the drop down (Up, Down, Toward camera, Away from camera). 

3.  Adjust the Detection Precision and Lost Precision parameters to set how closely 
    the hand must point in the chosen direction to trigger responses. The higher the
    precision values, the more closely the hand must point in the chosen direction.

4.  Check off Use Behavior to send custom triggers using the Behavior script. You
    will need to add the Behavior script to your project to use this option.

5.  Check off Call API Func to use custom scripting responses. Update the 
    OnDirectionDetectedLost script with your custom behavior or use the existing
    behavior and populate your own object in the Inspector panel. 

6.  Update or delete the hint to reflect your lens behavior.

*/
