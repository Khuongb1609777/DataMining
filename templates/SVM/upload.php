<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="X-UA-Compatible" content="ie=edge">
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.16.0/umd/popper.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.4.1/js/bootstrap.min.js"></script>
    <title>Upload File</title>
</head>
<body>
    <div>
            <form method="post" action ="SVM/data">
                <table class="table">
                    <tr>
                        <td>
                            Upload File
                        </td>
                        <td>
                            <input type="file" name="csvfile" id="csvfile">
                        </td>
                    </tr>

                    <tr>
                        <td colspan="2">
                            <input type="submit" name='submit' value="Upload">
                        </td>
                    </tr>
                </table>
                
            </form>
    </div>  

</body>
</html>