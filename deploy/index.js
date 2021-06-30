$(document).ready(function(){
    $("#ab").click(function(){

    const fileitems=$("#myFile").prop("files");
    console.log(fileitems)
    const input=$("#input");
    const output=$("#output");
    const file=fileitems[0].name
    input.attr("src",file)
    $("#bc").css("display","flex")
    $.ajax({
        type:'GET',
        url:'http://127.0.0.1:8081/'+file.toString(),
        success:function(res)
        {
            output.attr("src",res.filename)
        }
      })
    });
  });