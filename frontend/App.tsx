import React, { useState } from 'react';
import { Provider as PaperProvider } from 'react-native-paper';
import StartScreen from './src/screens/StartScreen';
import HomeScreen from './src/screens/HomeScreen';

export default function App() {
  const [started, setStarted] = useState(false);

  return (
    <PaperProvider>
      {started ? (
        <HomeScreen onBack={() => setStarted(false)} />
      ) : (
        <StartScreen onStart={() => setStarted(true)} />
      )}
    </PaperProvider>
  );
}
