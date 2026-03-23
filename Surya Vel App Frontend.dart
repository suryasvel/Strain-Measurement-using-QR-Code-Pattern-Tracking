/*
Code Name: App frontend coded in dart

Author: Surya S. Vel

Date: 2/17/2026

Version: 11
*/

// Get all the tools needed to build the app, pick photos, communicate through the internet, and handle files.
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart'; 
import 'package:http/http.dart' as http;        
import 'dart:io';
import 'dart:convert';

// Tells the app to run and show the main screen
void main() => runApp(const MaterialApp(
  debugShowCheckedModeBanner: false, // Hides the DEBUG banner in the corner
  home: ImageLabApp()
));

// Set up the main screen. Stateful means the screen can change while in use
class ImageLabApp extends StatefulWidget {
  const ImageLabApp({super.key});
  @override
  State<ImageLabApp> createState() => _ImageLabAppState();
}

// logic and layout for the screen
class _ImageLabAppState extends State<ImageLabApp> {
  // Variables that can change when app is running
  File? _selectedImage; // The picture the user picks
  String _status = "Select an image to begin"; // Message shown on screen
  String _qrInfo = "QR Data will show here"; // Message for the QR code data
  bool _isUploading = false; // True/false switch to know if currently loading

  // Opens the phone's photo gallery so the user can pick a picture
  Future<void> _pickImage() async {
    final picker = ImagePicker(); // Get the photo picker tool
    final pickedFile = await picker.pickImage(source: ImageSource.gallery); // Wait for user to pick something

    // If image is picked
    if (pickedFile != null) {
      // Update the screen with the selected image and new text
      setState(() {
        _selectedImage = File(pickedFile.path); // Save the picture
        _status = "Image selected! Ready to evaluate.";
        _qrInfo = "Awaiting analysis..."; 
      });
    }
  }

  // Send the image through the internet to a python backend
  Future<void> _sendToServer() async {
    // If no picture picked yet, do nothing
    if (_selectedImage == null) return;

    // Show loading screen
    setState(() {
      _isUploading = true;
      _status = "Analyzing Image...";
      _qrInfo = "Extracting Data...";
    });

    // If it crashes, catch the error
    try {
      // Set up the package to send to the python server
      var request = http.MultipartRequest(
        'POST', 
        Uri.parse('http://127.0.0.1:5000/compare') // Where to send image
      );

      // Put the picture in the package
      request.files.add(
        await http.MultipartFile.fromPath('image', _selectedImage!.path)
      );

      // Send the package and wait for a response
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      // "200" means the python server got the package
      if (response.statusCode == 200) {
        // Take the server's reply and extract data
        var responseData = json.decode(response.body);
        var results = responseData['data'];
        
        // If can't find strain or data, say error
        var strainValue = results['strain_micros'] ?? "N/A";
        var qrValue = results['qr_data'] ?? "No QR found by server";

        // Update the screen to show the final data
        setState(() {
          _status = "STRAIN: $strainValue micros";
          _qrInfo = "QR DATA: $qrValue";
        });
      } else {
        // If the server didn't work, show an error message
        setState(() => _status = "Server Error: Check Python terminal");
      }
    } catch (e) {
      // If couldn't connect to the server, show an error message
      setState(() => _status = "Connection Failed. Is Python running?");
    } finally {
      // Turn off the loading screen
      setState(() => _isUploading = false);
    }
  }

  // This part actually draws all the visual stuff on the screen.
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      // The top bar of the app
      appBar: AppBar(
        title: const Text("Evaluate Strain"),
        backgroundColor: Colors.blueGrey[900],
      ),
      // The main body of the app
      body: Center(
        // Makes the screen scrollable if needed
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20), // Add extra space aroundt the edges
          child: Column(
            children: [
              // If there is a picture, show it
              _selectedImage != null 
                ? ClipRRect(
                    borderRadius: BorderRadius.circular(10), // Round the corners
                    child: Image.file(_selectedImage!, height: 250, fit: BoxFit.cover),
                  ) 
                // If there is no picture, just show a big grey box.
                : Container(
                    height: 250, 
                    width: double.infinity,
                    color: Colors.grey[200],
                    child: const Icon(Icons.image, size: 80, color: Colors.grey),
                  ),
              
              const SizedBox(height: 20), // Empty space between items

              // Box that shows the QR code info.
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(15),
                decoration: BoxDecoration(
                  color: Colors.blue[50],
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.blue[100]!),
                ),
                child: Text(_qrInfo, style: const TextStyle(color: Colors.blue, fontWeight: FontWeight.bold)),
              ),

              const SizedBox(height: 15), // Empty space

              // White box that shows the status or strain result.
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(12),
                  boxShadow: const [BoxShadow(color: Colors.black12, blurRadius: 4)], // Adds a shadow
                ),
                child: Text(_status, textAlign: TextAlign.center, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, fontFamily: 'Courier')),
              ),

              const SizedBox(height: 30), // Empty space

              // Button to open the photo gallery.
              ElevatedButton.icon(
                onPressed: _pickImage, 
                icon: const Icon(Icons.photo_library), 
                label: const Text("Pick Image"),
                style: ElevatedButton.styleFrom(minimumSize: const Size(double.infinity, 50)),
              ),
              
              const SizedBox(height: 15), // Empty space

              // If loading, show a spinning circle
              _isUploading 
                ? const CircularProgressIndicator()
                // If not loading, show the green button to send the picture to the server.
                : ElevatedButton.icon(
                    // If no picture is picked yet, disable the button.
                    onPressed: _selectedImage == null ? null : _sendToServer, 
                    icon: const Icon(Icons.science), 
                    label: const Text("Run Analysis"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green[700], 
                      foregroundColor: Colors.white,
                      minimumSize: const Size(double.infinity, 50), // Makes the button wide and tall
                    ),
                  ),
            ],
          ),
        ),
      ),
    );
  }
}