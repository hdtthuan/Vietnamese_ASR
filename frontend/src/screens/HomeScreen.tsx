import React from "react";
import { View, StyleSheet } from "react-native";
import { Text, Button } from "react-native-paper";

type Props = {
  onBack: () => void;
};

export default function HomeScreen({ onBack }: Props) {
  return (
    <View style={styles.container}>
      <Text style={styles.title}>Home</Text>
      <Text style={styles.subtitle}>Trang Home — hiện chưa có chức năng nào.</Text>
      <Button mode="contained" onPress={onBack}>
        Back
      </Button>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#fff",
  },
  title: {
    fontSize: 28,
    fontWeight: "bold",
    marginBottom: 12,
  },
  subtitle: {
    fontSize: 16,
    color: "gray",
    marginBottom: 20,
    textAlign: "center",
  },
});