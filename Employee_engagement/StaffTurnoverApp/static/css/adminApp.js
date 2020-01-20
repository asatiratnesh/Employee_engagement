$(document).ready(function () {

	 var originalTextContent = 0;
	 var textContent = 0;
	 var updatedQuestions = 0;
	 var originalQuestions = 0;
	 var keywords = 0;
	 var userdatabase;
//admin checkboxes to select and deselect

	 $('.hasDatepicker').change(function(){
		 $("#jsonExport").hide();
	 });
	 $('#originalTextContent').change(function(){
		 $("#jsonExport").hide();

    if($(this).is(':checked')){
        originalTextContent=1;
    }
    else
    {
        originalTextContent=0;
    }

	});

	$('#textContent').change(function(){
		$("#jsonExport").hide();
    if($(this).is(':checked')){
        textContent = 1;
    }
    else
    {
        textContent = 0;
    }

	});

	$('#updatedQuestions').change(function(){
		$("#jsonExport").hide();
    if($(this).is(':checked')){
         updatedQuestions = 1;
    }
    else
    {
         updatedQuestions = 0;
    }

	});

	$('#originalQuestions').change(function(){
		$("#jsonExport").hide();
    if($(this).is(':checked')){
        originalQuestions = 1;
    }
    else
    {
        originalQuestions = 0;
    }

	});

	$('#keywords').change(function(){
		$("#jsonExport").hide();
    if($(this).is(':checked')){
        keywords = 1;
    }
    else
    {
        keywords = 0;
    }

	});


	//user submit data click which gets the user database entries

	 $(document).on('click', '#getUserData', function (e) {
		 e.preventDefault();
		 if ((Date.parse($("#datepicker1").val()) > Date.parse($("#datepicker2").val()))) {
        alert("End date should be greater than Start date");
    }
		else{

			$(".quillionzLoader").show();
	  var optionalUserData = {
		  "originalTextContent":originalTextContent,
		  "textContent":textContent,
		  "updatedQuestions":updatedQuestions,
		  "originalQuestions":originalQuestions,
		  "keywords":keywords
	  }
	  userDateData = {
            "toDate": $("#datepicker1").val(),
            "fromDate": $("#datepicker2").val(),
			"optionalUserData": optionalUserData
        }

        var userData = {
            "pythonApi": "getUserDatabase",
            "data": JSON.stringify(userDateData)
        };
        var myJSONString = JSON.stringify(userData);



		$.ajax({
            url: "https://localhost/RaptivityQA/API/Users/PythonAdminApi",
            data: myJSONString,
            type: 'post',
            contentType: "application/json",
            async: true,
            success: function (data) {
							$(".quillionzLoader").hide();
					userdatabase=JSON.parse(data);
					console.log(userdatabase);
					$("#jsonExport").show();
            },
            error: function (data) {
                console.log(data);
            }

        });

				}



	 });

	 $(document).on('click', '#jsonExport', function (e) {
			e.preventDefault();
			var filename = Date.now()+".json";
var jsonStr = JSON.stringify(userdatabase);
let element = document.createElement('a');
element.setAttribute('href', 'data:application/json;charset=utf-8,' + encodeURIComponent(jsonStr));
element.setAttribute('download', filename);
element.style.display = 'none';
document.body.appendChild(element);
element.click();

document.body.removeChild(element);
userdatabase=[];
			$(this).hide();


		});




});
