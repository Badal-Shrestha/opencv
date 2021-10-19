    /* JS comes here */


var width = 320; // We will scale the photo width to this
var height = 0; // This will be computed based on the input stream

var streaming = false;

var video = null;
var canvas = null;
var photo = null;
var startbutton = null;

function startup() {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    photo = document.getElementById('photo');
    startbutton = document.getElementById('startbutton');
    stopbutton = document.getElementById('stop-button');
    
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
        if (!streaming) {
            height = video.videoHeight / (video.videoWidth / width);

            if (isNaN(height)) {
                height = width / (4 / 3);
            }

            video.setAttribute('width', width);
            video.setAttribute('height', height);
            canvas.setAttribute('width', width);
            canvas.setAttribute('height', height);
            streaming = true;
        }
    }, false);

    startbutton.addEventListener('click', function(ev) {
        takepicture();
        ev.preventDefault();
    }, false);

    stopbutton.addEventListener('click',function(ev){
        localStream.getTracks().forEach( (track) => {
        track.stop();
    
        });
        

    })
    
    clearphoto();

    $("input[type=file]").change(function(){
        // alert($(this).val());
        var preview = document.getElementById('up-img');
        preview.src = window.URL.createObjectURL(this.files[0]);
        preview.style.display ="block";
        // $(this).src = window.URL.createObjectURL($(this).files[0]);
        
    });
    
    $("#upload-btn").click(function(){
        var form_data = new FormData($('#upload-image')[0]) // Creating form data
        var capture_selfi = photo.src // Captured img

        form_data.append("img2",capture_selfi) // Appending img on form data
        
        $.ajax({
            type: "POST",
            url: "/similarity", // Flask url for similarity calulation

            data : form_data,
            dataType : "json",
            contentType: false,
            cache : false,
            processData: false,
            beforeSend:function(){
                console.log("Send")
              },
            complete:function(data){ // Recive similarity score from the flask server
                console.log(data.responseText)
                response = data.responseText
                console.log(typeof(response))
                response = JSON.parse(response); // Parsing resonse
                var result = document.getElementById('result') 
                // displaying similarity score on pragraph Tag
                success = response['sucess']
                cid = response['cid']
                if (success){
                    window.location ="/userCIFUpdate/" + cid
                }else{
                    window.location = "/login"
                }

                result.innerText = response['score'][1]+ " score: " + response['score'][0] 
               

            },
            error: function (error,ajaxOptions, thrownError){
                alert("Error: " + error.status);
                console.log("Error. not working" , thrownError);
            }
        });
    });
}

// anotherbutton = document.getElementById('anotherbutton');
// anotherbutton.addEventListener('click',function(ev){
//         startup();
        

//     });
function clearphoto() {
    var context = canvas.getContext('2d');
    context.fillStyle = "#AAA";
    context.fillRect(0, 0, canvas.width, canvas.height);

    var data = canvas.toDataURL('image/png');
    photo.setAttribute('src', data);
}

function takepicture() {
    var context = canvas.getContext('2d');
    if (width && height) {
        canvas.width = width;
        canvas.height = height;
        context.drawImage(video, 0, 0, width, height);

        var data = canvas.toDataURL('image/png');
        photo.setAttribute('src', data);

        // canvas.setAttribute('src')
    } else {
        clearphoto();
    }
}

window.addEventListener('load', startup, false);
