    /* JS comes here */


let width = 600; // We will scale the photo width to this
let height = 480; // This will be computed based on the input stream

var streaming = false;

// var video = null;

let model
let canvas
let ctx
let video 
let direction
let frame_counter = 2
let head_direction = 'front'
let left
let front
let right
let interval
function startup() {
    left =[]
    front =[]
    right =[]
    video = document.getElementById('video');
    canvas =document.getElementById('canvas')
    direction =document.getElementById('direction')

    ctx = canvas.getContext("2d")
    
    navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false
        })
        .then(function(stream) {
            video.srcObject = stream;
            window.localStream = stream;
            video.play();

        })
        .catch(function(err) {
            console.log("An error occurred: " + err);
        });
        
    video.addEventListener('canplay', function(ev) {
        console.log(streaming)
        if (!streaming) {
            height = video.videoHeight / (video.videoWidth / width);

            if (isNaN(height)) {
                height = width / (4 / 3);
            }

            video.setAttribute('width', width);
            video.setAttribute('height', height);
            canvas.setAttribute('width', width);
            canvas.setAttribute('height', height);
            console.log(width,height)            
            streaming = true;


        }
        
    }, false);
    
    video.addEventListener('loadeddata',async ()=>{
        model = await blazeface.load();
        interval = setInterval(deteFaces, 100);
       
        // deteFaces();
    
    });
  

}


const deteFaces = async () => {
    const prediction = await model.estimateFaces(video,false)
    // console.log(prediction)
    ctx.drawImage(video,0,0,width,height)

    prediction.forEach((pred) => {
        pred.landmarks.forEach((landmark)=>{
            ctx.fillRect(landmark[0]-20,landmark[1]-10,5,5)
        });

        ctx.beginPath();
        ctx.lineWidth = "4";
        ctx.strokeStyle = "blue";
        x = Math.round(pred.topLeft[0])
        y = Math.round( pred.topLeft[1]) 
        w = Math.round(pred.bottomRight[0] - x) 
        h = Math.round(pred.bottomRight[1] - y) 
        ctx.rect(x,y,w ,h); //draw rect
        ctx.stroke();
        var form_data = new FormData() // Creating form data
        var data = canvas.toDataURL('image/jpg')
        var capture_selfi = data// Captured img

        form_data.append("video",capture_selfi) // Appending img on form data
        form_data.append("bounding_box",[x,y,w,h]) // Appending img on form data

        
        $.ajax({
            type: "POST",
            url: "/videoApproach", // Flask url for similarity calulation

            data : form_data,
            dataType : "json",
            contentType: false,
            cache : false,
            processData: false,
            beforeSend:function(){
                console.log("Send")
            },
            complete:function(data){ // Recive similarity score from the flask server
                response = JSON.parse(data.responseText);
                console.log(data.responseText,response[0],response[1])
                value = response[1]
                var label = parseInt(response[0])
                value = parseFloat(value)
                if (value > 0.5){
                    hpose = detect_pose(pred.landmarks)
                    

                    if (front.length < frame_counter){
                        front.push(label)
                    }
                    if (head_direction== "Turn left"  && hpose == 'left' && left.length < frame_counter ){
                        left.push(label)
                    }
                    if ( head_direction== "Turn right" && hpose == 'right' && right.length < frame_counter ){
                        right.push(label)
                    }

                    if (front.length == frame_counter){
                        head_direction = "Turn left"
                    }
                    if (left.length == frame_counter){
                        head_direction = "Turn right"
                    }
                    if (right.length == frame_counter){
                        head_direction = "Processing"
                    }
                    direction.innerHTML=  head_direction
                    
                }                // var result = document.getElementById('result') 
                // // displaying similarity score on pragraph Tag
                // result.innerText = response['score'][1]+ " score: " + response['score'][0] 
            

            },
            error: function (error,ajaxOptions, thrownError){
                alert("Error: " + error.status);
                console.log("Error. not working" , thrownError);
            }
        });
        

      
    });


    if (left.length == frame_counter && right.length == frame_counter && front.length == frame_counter){
        total_label = front.concat(left.concat(right))
        real_frame = total_label.filter(i => i === 1)
        fake_frame = total_label.filter(i => i === 0)
        if (real_frame > fake_frame){
            alert("Live Video")
            localStream.getTracks().forEach( (track) => {
                track.stop();
            });
            clearInterval(interval);
        }else {
            alert("Fake ");
            startup();
        }
        

    }

}
function eucDistance(a, b) {
    return a
        .map((x, i) => Math.abs( x - b[i] ) ** 2) // square the difference
        .reduce((sum, now) => sum + now) // sum
        ** (1/2)
}

function detect_pose(landmarks){
    var hpose =""
    var rEye = landmarks[0] 
    var lEye = landmarks[1] 
    var nose = landmarks[2]
    var mouth = landmarks[3]

    var mid = nose.map(function(num,idx){return (num +mouth[idx])/2 });
    // console.log(landmarks[0])
    // console.log("outside if",rEye[1]!=rEye[0])

    // if (true){
    var slope = (lEye[1]- rEye[1])/(lEye[0]-rEye[0])
    y_incpt = rEye[1] -(slope*rEye[0])

    // console.log("Inside if",y_incpt,rEye[1],slope,rEye )

    y = slope *mid[0] + y_incpt
    if (rEye[0] < mid[0] <lEye[0]){
        k1 = eucDistance(rEye,[mid[0],y])
        k2 = eucDistance([mid[0],y],lEye)
        k3 = eucDistance([mid[0],nose[1]],[mid[0],mouth[1]])
        k4 = eucDistance([mid[0],nose[1]],[mid[0],y])
        // console.log("K22", k1,k2,k3,4)/
        if ((k2/k1)  <=0.5){
            hpose = 'left'
            // direction.innerHTML= "Left"
        }else if ((k1/k2) <=0.5){
            hpose = 'right'

            // direction.innerHTML= "Right"

        }



    }
    // }

    return hpose
    
}


window.addEventListener('load', startup, false);
// startup()