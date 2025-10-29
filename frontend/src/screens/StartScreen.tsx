import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Button, Text } from 'react-native-paper';

type Props = {
  onStart: () => void;
};

export default function StartScreen({ onStart }: Props) {
  return (
    <View style={styles.container}>
      <Text variant="headlineMedium" style={styles.title}>
        Accent Detector
      </Text>
      <Text variant="bodyLarge" style={styles.subtitle}>
        Welcome — detect regional accents from audio samples.
      </Text>
      <Button mode="contained" onPress={onStart} style={styles.button}>
        Start
      </Button>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  title: {
    marginBottom: 8,
  },
  subtitle: {
    color: '#555',
  },
  button: {
    marginTop: 20,
    width: 140,
  },
});