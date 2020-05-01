<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
    <title>Register now</title>
    <link href="//maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet" id="bootstrap-css">
    <link rel='stylesheet' href='registercss.css' type='text/css'>
    <script src="//maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js"></script>
    <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<!------ Include the above in your HEAD tag ---------->
</head>
<body>


<section class="register-block">
  <div class="container">
	  <div class="row">
		  <div class="col-md-4 register-sec">
		    <h2 class="text-center">Register Now</h2>
      
        <form class="register-form" action="checkregister.php" onsubmit="return checkregister();">


          <div class="form-group">
            <label for="exampleInputName1" id="name" class="text-uppercase">Name <a style="color:red">*</a></label>
            <input type="name" class="form-control" placeholder="Your name">
            
          </div>


          <div class="form-group">
            <label for="exampleInputAddress1" id="address" class="text-uppercase">Address</label>
            <input type="address1" class="form-control" placeholder="Address">
          </div>

          
          <div class="form-group">
            <label for="exampleInputTown1" id="town" class="text-uppercase">Town </label>
            <input type="town" class="form-control" placeholder="Town">
          </div>


          <div class="form-group">
            <label for="exampleInputCountry1" id="country" class="text-uppercase">Country </label>
            <input type="country" class="form-control" placeholder="Country">
          </div>



          <div class="form-group">
            <label for="exampleInputUsername" id="username" class="text-uppercase">Username <a style="color:red">*</a></label>
            <input type="text" class="form-control" placeholder="Username">      
          </div>


          <div class="form-group">
            <label for="exampleInputEmail" id="email" class="text-uppercase">Email <a style="color:red">*</a></label>
            <input type="email" class="form-control" placeholder="Your email">
          </div>


          <div class="form-group">
            <label for="exampleInputPassword1" id="password" class="text-uppercase">Password <a style="color:red">*</a></label>
            <input type="password" class="form-control" placeholder="Password">
          </div>
                    
                    
          <div class="form-check">
            <button type="submit" class="btn btn-register float-right">Submit</button>
          </div>


          </form>
		      </div>
		      <div class="col-md-8 banner-sec">
      </div>
	  </div>
  </div>
</section>


<script>
  function checkregister(){

    var name = document.getElementById('name').value;
    var address = document.getElementById('address').value;
    var town = document.getElementById('town').value;
    var country = document.getElementById('country').value;
    var username = document.getElementById('username').value;
    var email = document.getElementById('email').value;
    var password = document.getElementById('password').value;    
    
    var loi = "";
    if(name ==""){
      loi += "";
    }

    if (password ==""){
      loi += "Mật khẩu không được để trống\n";
    }

    if(avatar ==""){
      loi += "Bạn chưa chọn hình đại diện\n";
    }

    if(gender == ""){
      loi += "Bạn chưa chọn giới tính \n"
    }

    if(nghenghiep ==""){
      loi += "Bạn chưa chọn nghề nghiệp \n";
    }

    if (resultst == ""){
      loi += "Bạn chưa chọn sở thích\n"
    }

    if(!REusername.test(username)){
      loi += "Tên đăng nhập phải bắt đầu bằng chữ cái, theo sau có thể là chữ cái hoặc là số, chiều dài tối thiểu 6 và tối đa 15 ký tự. \n";
    }
    if(!REpass.test(password)){
      loi += "Mật khẩu phải xuất hiện cả chữ và số, dài tối thiểu 6 và tối đa là 15 ký tự\n";
    }
    if (password != password2){
      loi += "Mật khẩu và mật khẩu dòng 2 phải khớp nhau\n";
    }
    if(loi==""){
      return true;
      
    }else{
      alert (loi);
      return false;
      
    }


  }
</script>



    
</body>
</html>